import sys
_module = sys.modules[__name__]
del sys
data = _module
coco = _module
config = _module
data_augment = _module
layers = _module
functions = _module
detection = _module
prior_box = _module
modules = _module
l2norm = _module
multibox_loss = _module
LRF_COCO_300 = _module
LRF_COCO_512 = _module
models = _module
test_LRF = _module
utils = _module
box_utils = _module
build = _module
nms = _module
py_cpu_nms = _module
nms_wrapper = _module
pycocotools = _module
cocoeval = _module
mask = _module
timer = _module

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


import torch


import torch.utils.data as data


import torchvision.transforms as transforms


import numpy as np


from torchvision import transforms


import random


import math


import torch.nn as nn


import torch.backends.cudnn as cudnn


from torch.autograd import Function


from torch.autograd import Variable


from math import sqrt as sqrt


from itertools import product as product


import torch.nn.init as init


import torch.nn.functional as F


from collections import OrderedDict


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


GPU = False


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
        loc_data, conf_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = priors.size(0)
        num_classes = self.num_classes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, (-1)].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if GPU:
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
        loss_c[pos.view(-1)] = 0
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


class LDS(nn.Module):

    def __init__(self):
        super(LDS, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

    def forward(self, x):
        x_pool1 = self.pool1(x)
        x_pool2 = self.pool2(x_pool1)
        x_pool3 = self.pool3(x_pool2)
        return x_pool3


class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(ConvBlock, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=False) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class LSN_init(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(LSN_init, self).__init__()
        self.out_channels = out_planes
        inter_planes = out_planes // 4
        self.part_a = nn.Sequential(ConvBlock(in_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=1), ConvBlock(inter_planes, inter_planes, kernel_size=1, stride=1), ConvBlock(inter_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=1))
        self.part_b = ConvBlock(inter_planes, out_planes, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        out1 = self.part_a(x)
        out2 = self.part_b(out1)
        return out1, out2


class LSN_later(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(LSN_later, self).__init__()
        self.out_channels = out_planes
        inter_planes = out_planes // 4
        self.part_a = ConvBlock(in_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=1)
        self.part_b = ConvBlock(inter_planes, out_planes, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        out1 = self.part_a(x)
        out2 = self.part_b(out1)
        return out1, out2


class IBN(nn.Module):

    def __init__(self, out_planes, bn=True):
        super(IBN, self).__init__()
        self.out_channels = out_planes
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True) if bn else None

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        return x


class One_Three_Conv(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(One_Three_Conv, self).__init__()
        self.out_channels = out_planes
        inter_planes = in_planes // 4
        self.single_branch = nn.Sequential(ConvBlock(in_planes, inter_planes, kernel_size=1, stride=1), ConvBlock(inter_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=1, relu=False))

    def forward(self, x):
        out = self.single_branch(x)
        return out


class Relu_Conv(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super(Relu_Conv, self).__init__()
        self.out_channels = out_planes
        self.relu = nn.ReLU(inplace=False)
        self.single_branch = nn.Sequential(ConvBlock(in_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=1))

    def forward(self, x):
        x = self.relu(x)
        out = self.single_branch(x)
        return out


class Ds_Conv(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, padding=(1, 1)):
        super(Ds_Conv, self).__init__()
        self.out_channels = out_planes
        self.single_branch = nn.Sequential(ConvBlock(in_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=padding, relu=False))

    def forward(self, x):
        out = self.single_branch(x)
        return out


class LRFNet(nn.Module):
    """LRFNet for object detection
    The network is based on the SSD architecture.
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(LRFNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size
        self.base = nn.ModuleList(base)
        self.lds = LDS()
        self.Norm1 = Relu_Conv(512, 512, stride=1)
        self.Norm2 = Relu_Conv(1024, 1024, stride=1)
        self.Norm3 = Relu_Conv(512, 512, stride=1)
        self.Norm4 = Relu_Conv(256, 256, stride=1)
        self.Norm5 = Relu_Conv(256, 256, stride=1)
        self.icn1 = LSN_init(3, 512, stride=1)
        self.icn2 = LSN_later(128, 1024, stride=2)
        self.icn3 = LSN_later(256, 512, stride=2)
        self.icn4 = LSN_later(128, 256, stride=2)
        self.dsc1 = Ds_Conv(512, 1024, stride=2, padding=(1, 1))
        self.dsc2 = Ds_Conv(1024, 512, stride=2, padding=(1, 1))
        self.dsc3 = Ds_Conv(512, 256, stride=2, padding=(1, 1))
        self.dsc4 = Ds_Conv(256, 256, stride=2, padding=(1, 1))
        self.agent1 = ConvBlock(512, 256, kernel_size=1, stride=1)
        self.agent2 = ConvBlock(1024, 512, kernel_size=1, stride=1)
        self.agent3 = ConvBlock(512, 256, kernel_size=1, stride=1)
        self.agent4 = ConvBlock(256, 128, kernel_size=1, stride=1)
        self.proj1 = ConvBlock(1024, 128, kernel_size=1, stride=1)
        self.proj2 = ConvBlock(512, 128, kernel_size=1, stride=1)
        self.proj3 = ConvBlock(256, 128, kernel_size=1, stride=1)
        self.proj4 = ConvBlock(256, 128, kernel_size=1, stride=1)
        self.convert1 = ConvBlock(512, 256, kernel_size=1)
        self.convert2 = ConvBlock(384, 512, kernel_size=1)
        self.convert3 = ConvBlock(256, 256, kernel_size=1)
        self.convert4 = ConvBlock(128, 128, kernel_size=1)
        self.merge1 = ConvBlock(512, 512, kernel_size=3, stride=1, padding=1)
        self.merge2 = ConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.merge3 = ConvBlock(512, 512, kernel_size=3, stride=1, padding=1)
        self.merge4 = ConvBlock(256, 256, kernel_size=3, stride=1, padding=1)
        self.ibn1 = IBN(512, bn=True)
        self.ibn2 = IBN(1024, bn=True)
        self.relu = nn.ReLU(inplace=False)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax()

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,512,512].

        Return:
            Depending on phase:
            test:
                list of concat outputs from:
                    1: softmax layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        new_sources = list()
        x_pool = self.lds(x)
        for k in range(22):
            x = self.base[k](x)
        conv4_3_bn = self.ibn1(x)
        x_pool1_skip, x_pool1_icn = self.icn1(x_pool)
        s = self.Norm1(conv4_3_bn * x_pool1_icn)
        for k in range(22, 34):
            x = self.base[k](x)
        conv7_bn = self.ibn2(x)
        x_pool2_skip, x_pool2_icn = self.icn2(x_pool1_skip)
        p = self.Norm2(self.dsc1(s) + conv7_bn * x_pool2_icn)
        x = self.base[34](x)
        for k, v in enumerate(self.extras):
            x = v(x)
            if k == 0:
                x_pool3_skip, x_pool3_icn = self.icn3(x_pool2_skip)
                w = self.Norm3(self.dsc2(p) + x * x_pool3_icn)
            elif k == 2:
                x_pool4_skip, x_pool4_icn = self.icn4(x_pool3_skip)
                q = self.Norm4(self.dsc3(w) + x * x_pool4_icn)
            elif k == 4:
                o = self.Norm5(self.dsc4(q) + x)
                sources.append(o)
            elif k == 7 or k == 9:
                sources.append(x)
            else:
                pass
        tmp1 = self.proj1(p)
        tmp2 = self.proj2(w)
        tmp3 = self.proj3(q)
        tmp4 = self.proj4(o)
        proj1 = F.upsample(tmp1, scale_factor=2, mode='bilinear')
        proj2 = F.upsample(tmp2, scale_factor=4, mode='bilinear')
        proj3 = F.upsample(tmp3, scale_factor=8, mode='bilinear')
        proj4 = F.upsample(tmp4, scale_factor=16, mode='bilinear')
        proj = torch.cat([proj1, proj2, proj3, proj4], dim=1)
        agent1 = self.agent1(s)
        convert1 = self.convert1(proj)
        pred1 = torch.cat([agent1, convert1], dim=1)
        pred1 = self.merge1(pred1)
        new_sources.append(pred1)
        proj2 = F.upsample(tmp2, scale_factor=2, mode='bilinear')
        proj3 = F.upsample(tmp3, scale_factor=4, mode='bilinear')
        proj4 = F.upsample(tmp4, scale_factor=8, mode='bilinear')
        proj = torch.cat([proj2, proj3, proj4], dim=1)
        agent2 = self.agent2(p)
        convert2 = self.convert2(proj)
        pred2 = torch.cat([agent2, convert2], dim=1)
        pred2 = self.merge2(pred2)
        new_sources.append(pred2)
        proj3 = F.upsample(tmp3, scale_factor=2, mode='bilinear')
        proj4 = F.upsample(tmp4, scale_factor=4, mode='bilinear')
        proj = torch.cat([proj3, proj4], dim=1)
        agent3 = self.agent3(w)
        convert3 = self.convert3(proj)
        pred3 = torch.cat([agent3, convert3], dim=1)
        pred3 = self.merge3(pred3)
        new_sources.append(pred3)
        proj4 = F.upsample(tmp4, scale_factor=2, mode='bilinear')
        proj = proj4
        agent4 = self.agent4(q)
        convert4 = self.convert4(proj)
        pred4 = torch.cat([agent4, convert4], dim=1)
        pred4 = self.merge4(pred4)
        new_sources.append(pred4)
        for prediction in sources:
            new_sources.append(prediction)
        for x, l, c in zip(new_sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == 'test':
            output = loc.view(loc.size(0), -1, 4), self.softmax(conf.view(-1, self.num_classes))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes)
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            None
            self.load_state_dict(torch.load(base_file))
            None
        else:
            None


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Ds_Conv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IBN,
     lambda: ([], {'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L2Norm,
     lambda: ([], {'n_channels': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LDS,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (LSN_init,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LSN_later,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (One_Three_Conv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Relu_Conv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_vaesl_LRF_Net(_paritybench_base):
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

