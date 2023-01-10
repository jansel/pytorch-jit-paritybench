import sys
_module = sys.modules[__name__]
del sys
data = _module
config = _module
widerface = _module
demo = _module
layers = _module
bbox_utils = _module
functions = _module
detection = _module
prior_box = _module
modules = _module
l2norm = _module
multibox_loss = _module
prepare_wider_data = _module
pyramidbox = _module
afw_test = _module
fddb_test = _module
pascal_test = _module
wider_test = _module
train = _module
utils = _module
augmentations = _module

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


import random


import torch.nn as nn


import torch.backends.cudnn as cudnn


from torch.autograd import Variable


import time


from torch.autograd import Function


from itertools import product as product


import math


import torch.nn.init as init


import torch.nn.functional as F


import torchvision.transforms as transforms


import scipy.io as sio


import torch.optim as optim


from torchvision import transforms


import types


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
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip(max_xy - min_xy, a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


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


def match_ssd(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
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

    def __init__(self, cfg, use_gpu=True, use_head_loss=False):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = cfg.NUM_CLASSES
        self.negpos_ratio = cfg.NEG_POS_RATIOS
        self.variance = cfg.VARIANCE
        self.use_head_loss = use_head_loss
        self.threshold = cfg.FACE.OVERLAP_THRESH
        self.match = match_ssd

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        if self.use_head_loss:
            _, _, loc_data, conf_data, priors = predictions
        else:
            loc_data, conf_data, _, _, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = priors.size(0)
        num_classes = self.num_classes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            self.match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t
            conf_t = conf_t
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
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
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        N = num_pos.data.sum() if num_pos.data.sum() > 0 else num
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c


class conv_bn(nn.Module):
    """docstring for conv"""

    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(conv_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_plane, out_plane, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_plane)

    def forward(self, x):
        x = self.conv1(x)
        return self.bn1(x)


class CPM(nn.Module):
    """docstring for CPM"""

    def __init__(self, in_plane):
        super(CPM, self).__init__()
        self.branch1 = conv_bn(in_plane, 1024, 1, 1, 0)
        self.branch2a = conv_bn(in_plane, 256, 1, 1, 0)
        self.branch2b = conv_bn(256, 256, 3, 1, 1)
        self.branch2c = conv_bn(256, 1024, 1, 1, 0)
        self.ssh_1 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.ssh_dimred = nn.Conv2d(1024, 128, kernel_size=3, stride=1, padding=1)
        self.ssh_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.ssh_3a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.ssh_3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out_residual = self.branch1(x)
        x = F.relu(self.branch2a(x), inplace=True)
        x = F.relu(self.branch2b(x), inplace=True)
        x = self.branch2c(x)
        rescomb = F.relu(x + out_residual, inplace=True)
        ssh1 = self.ssh_1(rescomb)
        ssh_dimred = F.relu(self.ssh_dimred(rescomb), inplace=True)
        ssh_2 = self.ssh_2(ssh_dimred)
        ssh_3a = F.relu(self.ssh_3a(ssh_dimred), inplace=True)
        ssh_3b = self.ssh_3b(ssh_3a)
        ssh_out = torch.cat([ssh1, ssh_2, ssh_3b], dim=1)
        ssh_out = F.relu(ssh_out, inplace=True)
        return ssh_out


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        rem_areas = torch.index_select(area, 0, idx)
        union = rem_areas - inter + area[i]
        IoU = inter / union
        idx = idx[IoU.le(overlap)]
    return keep, count


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, cfg):
        self.num_classes = cfg.NUM_CLASSES
        self.top_k = cfg.TOP_K
        self.nms_thresh = cfg.NMS_THRESH
        self.conf_thresh = cfg.CONF_THRESH
        self.variance = cfg.VARIANCE

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4] 
        """
        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
        batch_priors = prior_data.view(-1, num_priors, 4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)
        decoded_boxes = decode(loc_data.view(-1, 4), batch_priors, self.variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        for i in range(num):
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes_ = boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes_, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes_[ids[:count]]), 1)
        return output


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, input_size, feature_maps, cfg):
        super(PriorBox, self).__init__()
        self.imh = input_size[0]
        self.imw = input_size[1]
        self.variance = cfg.VARIANCE or [0.1]
        self.min_sizes = cfg.ANCHOR_SIZES
        self.steps = cfg.STEPS
        self.clip = cfg.CLIP
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')
        self.feature_maps = feature_maps

    def forward(self):
        mean = []
        for k in range(len(self.feature_maps)):
            feath = self.feature_maps[k][0]
            featw = self.feature_maps[k][1]
            for i, j in product(range(feath), range(featw)):
                f_kw = self.imw / self.steps[k]
                f_kh = self.imh / self.steps[k]
                cx = (j + 0.5) / f_kw
                cy = (i + 0.5) / f_kh
                s_kw = self.min_sizes[k] / self.imw
                s_kh = self.min_sizes[k] / self.imh
                mean += [cx, cy, s_kw, s_kh]
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class PyramidBox(nn.Module):
    """docstring for PyramidBox"""

    def __init__(self, phase, base, extras, lfpn_cpm, head, num_classes):
        super(PyramidBox, self).__init__()
        self.vgg = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.L2Norm3_3 = L2Norm(256, 10)
        self.L2Norm4_3 = L2Norm(512, 8)
        self.L2Norm5_3 = L2Norm(512, 5)
        self.lfpn_topdown = nn.ModuleList(lfpn_cpm[0])
        self.lfpn_later = nn.ModuleList(lfpn_cpm[1])
        self.cpm = nn.ModuleList(lfpn_cpm[2])
        self.loc_layers = nn.ModuleList(head[0])
        self.conf_layers = nn.ModuleList(head[1])
        self.is_infer = False
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)
            self.is_infer = True

    def _upsample_prod(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') * y

    def forward(self, x):
        size = x.size()[2:]
        for k in range(16):
            x = self.vgg[k](x)
        conv3_3 = x
        for k in range(16, 23):
            x = self.vgg[k](x)
        conv4_3 = x
        for k in range(23, 30):
            x = self.vgg[k](x)
        conv5_3 = x
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        convfc_7 = x
        for k in range(2):
            x = F.relu(self.extras[k](x), inplace=True)
        conv6_2 = x
        for k in range(2, 4):
            x = F.relu(self.extras[k](x), inplace=True)
        conv7_2 = x
        x = F.relu(self.lfpn_topdown[0](convfc_7), inplace=True)
        lfpn2_on_conv5 = F.relu(self._upsample_prod(x, self.lfpn_later[0](conv5_3)), inplace=True)
        x = F.relu(self.lfpn_topdown[1](lfpn2_on_conv5), inplace=True)
        lfpn1_on_conv4 = F.relu(self._upsample_prod(x, self.lfpn_later[1](conv4_3)), inplace=True)
        x = F.relu(self.lfpn_topdown[2](lfpn1_on_conv4), inplace=True)
        lfpn0_on_conv3 = F.relu(self._upsample_prod(x, self.lfpn_later[2](conv3_3)), inplace=True)
        ssh_conv3_norm = self.cpm[0](self.L2Norm3_3(lfpn0_on_conv3))
        ssh_conv4_norm = self.cpm[1](self.L2Norm4_3(lfpn1_on_conv4))
        ssh_conv5_norm = self.cpm[2](self.L2Norm5_3(lfpn2_on_conv5))
        ssh_convfc7 = self.cpm[3](convfc_7)
        ssh_conv6 = self.cpm[4](conv6_2)
        ssh_conv7 = self.cpm[5](conv7_2)
        face_locs, face_confs = [], []
        head_locs, head_confs = [], []
        N = ssh_conv3_norm.size(0)
        mbox_loc = self.loc_layers[0](ssh_conv3_norm)
        face_loc, head_loc = torch.chunk(mbox_loc, 2, dim=1)
        face_loc = face_loc.permute(0, 2, 3, 1).contiguous().view(N, -1, 4)
        if not self.is_infer:
            head_loc = head_loc.permute(0, 2, 3, 1).contiguous().view(N, -1, 4)
        mbox_conf = self.conf_layers[0](ssh_conv3_norm)
        face_conf1 = mbox_conf[:, 3:4, :, :]
        face_conf3_maxin, _ = torch.max(mbox_conf[:, 0:3, :, :], dim=1, keepdim=True)
        face_conf = torch.cat((face_conf3_maxin, face_conf1), dim=1)
        face_conf = face_conf.permute(0, 2, 3, 1).contiguous().view(N, -1, 2)
        if not self.is_infer:
            head_conf3_maxin, _ = torch.max(mbox_conf[:, 4:7, :, :], dim=1, keepdim=True)
            head_conf1 = mbox_conf[:, 7:, :, :]
            head_conf = torch.cat((head_conf3_maxin, head_conf1), dim=1)
            head_conf = head_conf.permute(0, 2, 3, 1).contiguous().view(N, -1, 2)
        face_locs.append(face_loc)
        face_confs.append(face_conf)
        if not self.is_infer:
            head_locs.append(head_loc)
            head_confs.append(head_conf)
        inputs = [ssh_conv4_norm, ssh_conv5_norm, ssh_convfc7, ssh_conv6, ssh_conv7]
        feature_maps = []
        feat_size = ssh_conv3_norm.size()[2:]
        feature_maps.append([feat_size[0], feat_size[1]])
        for i, feat in enumerate(inputs):
            feat_size = feat.size()[2:]
            feature_maps.append([feat_size[0], feat_size[1]])
            mbox_loc = self.loc_layers[i + 1](feat)
            face_loc, head_loc = torch.chunk(mbox_loc, 2, dim=1)
            face_loc = face_loc.permute(0, 2, 3, 1).contiguous().view(N, -1, 4)
            if not self.is_infer:
                head_loc = head_loc.permute(0, 2, 3, 1).contiguous().view(N, -1, 4)
            mbox_conf = self.conf_layers[i + 1](feat)
            face_conf1 = mbox_conf[:, 0:1, :, :]
            face_conf3_maxin, _ = torch.max(mbox_conf[:, 1:4, :, :], dim=1, keepdim=True)
            face_conf = torch.cat((face_conf1, face_conf3_maxin), dim=1)
            face_conf = face_conf.permute(0, 2, 3, 1).contiguous().view(N, -1, 2)
            if not self.is_infer:
                head_conf = mbox_conf[:, 4:, :, :].permute(0, 2, 3, 1).contiguous().view(N, -1, 2)
            face_locs.append(face_loc)
            face_confs.append(face_conf)
            if not self.is_infer:
                head_locs.append(head_loc)
                head_confs.append(head_conf)
        face_mbox_loc = torch.cat(face_locs, dim=1)
        face_mbox_conf = torch.cat(face_confs, dim=1)
        if not self.is_infer:
            head_mbox_loc = torch.cat(head_locs, dim=1)
            head_mbox_conf = torch.cat(head_confs, dim=1)
        priors_boxes = PriorBox(size, feature_maps, cfg)
        priors = Variable(priors_boxes.forward(), volatile=True)
        if not self.is_infer:
            output = face_mbox_loc, face_mbox_conf, head_mbox_loc, head_mbox_conf, priors
        else:
            output = self.detect(face_mbox_loc, self.softmax(face_mbox_conf), priors)
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            None
            mdata = torch.load(base_file, map_location=lambda storage, loc: storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            None
        else:
            None
        return epoch

    def xavier(self, param):
        init.xavier_uniform(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            if 'bias' in m.state_dict().keys():
                m.bias.data.zero_()
        if isinstance(m, nn.ConvTranspose2d):
            self.xavier(m.weight.data)
            if 'bias' in m.state_dict().keys():
                m.bias.data.zero_()
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data[...] = 1
            m.bias.data.zero_()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CPM,
     lambda: ([], {'in_plane': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L2Norm,
     lambda: ([], {'n_channels': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (conv_bn,
     lambda: ([], {'in_plane': 4, 'out_plane': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_yxlijun_Pyramidbox_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

