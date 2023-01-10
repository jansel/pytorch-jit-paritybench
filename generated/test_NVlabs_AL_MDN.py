import sys
_module = sys.modules[__name__]
del sys
active_learning_loop = _module
data = _module
coco = _module
coco_eval = _module
config = _module
voc0712 = _module
demo = _module
eval_coco = _module
eval_voc = _module
layers = _module
box_utils = _module
functions = _module
detection_gmm = _module
prior_box = _module
modules = _module
l2norm = _module
multibox_loss_gmm = _module
ssd_gmm = _module
subset_sequential_sampler = _module
train_ssd_gmm_active_learining = _module
train_ssd_gmm_supervised_learning = _module
utils = _module
augmentations = _module
test_voc = _module

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


import time


import torch


import torch.nn.functional as F


import torch.nn as nn


import torch.optim as optim


import torch.backends.cudnn as cudnn


import torch.nn.init as init


import torch.utils.data as data


import numpy as np


import math


import torchvision.transforms as transforms


from torch.autograd import Variable


from matplotlib import pyplot as plt


from torch.autograd import Function


from math import sqrt as sqrt


from itertools import product as product


from torch.utils.data.sampler import Sampler


import random


from torch.utils.data.sampler import SubsetRandomSampler


from torch.utils.data.sampler import SequentialSampler


from torchvision import transforms


import types


from numpy import random


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


def Gaussian(y, mu, var):
    eps = 0.3
    result = (y - mu) / var
    result = result ** 2 / 2 * -1
    exp = torch.exp(result)
    result = exp / math.sqrt(2 * math.pi) / (var + eps)
    return result


def NLL_loss(bbox_gt, bbox_pred, bbox_var):
    bbox_var = torch.sigmoid(bbox_var)
    prob = Gaussian(bbox_gt, bbox_pred, bbox_var)
    return prob


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
    g_wh = torch.log(g_wh + 1e-09) / variances[1]
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
    conf = labels[best_truth_idx] + 1
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf


class MultiBoxLoss_GMM(nn.Module):
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
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, use_gpu=True, cls_type='Type-1'):
        super(MultiBoxLoss_GMM, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.cls_type = cls_type

    def forward(self, predictions, targets):
        priors, loc_mu_1, loc_var_1, loc_pi_1, loc_mu_2, loc_var_2, loc_pi_2, loc_mu_3, loc_var_3, loc_pi_3, loc_mu_4, loc_var_4, loc_pi_4, conf_mu_1, conf_var_1, conf_pi_1, conf_mu_2, conf_var_2, conf_pi_2, conf_mu_3, conf_var_3, conf_pi_3, conf_mu_4, conf_var_4, conf_pi_4 = predictions
        num = loc_mu_1.size(0)
        priors = priors[:loc_mu_1.size(1), :]
        num_priors = priors.size(0)
        num_classes = self.num_classes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t
            conf_t = conf_t
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_mu_1)
        loc_mu_1_ = loc_mu_1[pos_idx].view(-1, 4)
        loc_mu_2_ = loc_mu_2[pos_idx].view(-1, 4)
        loc_mu_3_ = loc_mu_3[pos_idx].view(-1, 4)
        loc_mu_4_ = loc_mu_4[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l_1 = NLL_loss(loc_t, loc_mu_1_, loc_var_1[pos_idx].view(-1, 4))
        loss_l_2 = NLL_loss(loc_t, loc_mu_2_, loc_var_2[pos_idx].view(-1, 4))
        loss_l_3 = NLL_loss(loc_t, loc_mu_3_, loc_var_3[pos_idx].view(-1, 4))
        loss_l_4 = NLL_loss(loc_t, loc_mu_4_, loc_var_4[pos_idx].view(-1, 4))
        loc_pi_1_ = loc_pi_1[pos_idx].view(-1, 4)
        loc_pi_2_ = loc_pi_2[pos_idx].view(-1, 4)
        loc_pi_3_ = loc_pi_3[pos_idx].view(-1, 4)
        loc_pi_4_ = loc_pi_4[pos_idx].view(-1, 4)
        pi_all = torch.stack([loc_pi_1_.reshape(-1), loc_pi_2_.reshape(-1), loc_pi_3_.reshape(-1), loc_pi_4_.reshape(-1)])
        pi_all = pi_all.transpose(0, 1)
        pi_all = torch.softmax(pi_all, dim=1).transpose(0, 1).reshape(-1)
        loc_pi_1_, loc_pi_2_, loc_pi_3_, loc_pi_4_ = torch.split(pi_all, loc_pi_1_.reshape(-1).size(0), dim=0)
        loc_pi_1_ = loc_pi_1_.view(-1, 4)
        loc_pi_2_ = loc_pi_2_.view(-1, 4)
        loc_pi_3_ = loc_pi_3_.view(-1, 4)
        loc_pi_4_ = loc_pi_4_.view(-1, 4)
        _loss_l = loc_pi_1_ * loss_l_1 + loc_pi_2_ * loss_l_2 + loc_pi_3_ * loss_l_3 + loc_pi_4_ * loss_l_4
        epsi = 10 ** -9
        balance = 2.0
        loss_l = -torch.log(_loss_l + epsi) / balance
        loss_l = loss_l.sum()
        if self.cls_type == 'Type-1':
            conf_pi_1_ = conf_pi_1.view(-1, 1)
            conf_pi_2_ = conf_pi_2.view(-1, 1)
            conf_pi_3_ = conf_pi_3.view(-1, 1)
            conf_pi_4_ = conf_pi_4.view(-1, 1)
            conf_pi_all = torch.stack([conf_pi_1_.reshape(-1), conf_pi_2_.reshape(-1), conf_pi_3_.reshape(-1), conf_pi_4_.reshape(-1)])
            conf_pi_all = conf_pi_all.transpose(0, 1)
            conf_pi_all = torch.softmax(conf_pi_all, dim=1).transpose(0, 1).reshape(-1)
            conf_pi_1_, conf_pi_2_, conf_pi_3_, conf_pi_4_ = torch.split(conf_pi_all, conf_pi_1_.reshape(-1).size(0), dim=0)
            conf_pi_1_ = conf_pi_1_.view(conf_pi_1.size(0), -1)
            conf_pi_2_ = conf_pi_2_.view(conf_pi_2.size(0), -1)
            conf_pi_3_ = conf_pi_3_.view(conf_pi_3.size(0), -1)
            conf_pi_4_ = conf_pi_4_.view(conf_pi_4.size(0), -1)
            conf_var_1 = torch.sigmoid(conf_var_1)
            conf_var_2 = torch.sigmoid(conf_var_2)
            conf_var_3 = torch.sigmoid(conf_var_3)
            conf_var_4 = torch.sigmoid(conf_var_4)
            rand_val_1 = torch.randn(conf_var_1.size(0), conf_var_1.size(1), conf_var_1.size(2))
            rand_val_2 = torch.randn(conf_var_2.size(0), conf_var_2.size(1), conf_var_2.size(2))
            rand_val_3 = torch.randn(conf_var_3.size(0), conf_var_3.size(1), conf_var_3.size(2))
            rand_val_4 = torch.randn(conf_var_4.size(0), conf_var_4.size(1), conf_var_4.size(2))
            batch_conf_1 = (conf_mu_1 + torch.sqrt(conf_var_1) * rand_val_1).view(-1, self.num_classes)
            batch_conf_2 = (conf_mu_2 + torch.sqrt(conf_var_2) * rand_val_2).view(-1, self.num_classes)
            batch_conf_3 = (conf_mu_3 + torch.sqrt(conf_var_3) * rand_val_3).view(-1, self.num_classes)
            batch_conf_4 = (conf_mu_4 + torch.sqrt(conf_var_4) * rand_val_4).view(-1, self.num_classes)
            loss_c_1 = log_sum_exp(batch_conf_1) - batch_conf_1.gather(1, conf_t.view(-1, 1))
            loss_c_2 = log_sum_exp(batch_conf_2) - batch_conf_2.gather(1, conf_t.view(-1, 1))
            loss_c_3 = log_sum_exp(batch_conf_3) - batch_conf_3.gather(1, conf_t.view(-1, 1))
            loss_c_4 = log_sum_exp(batch_conf_4) - batch_conf_4.gather(1, conf_t.view(-1, 1))
            loss_c = loss_c_1 * conf_pi_1_.view(-1, 1) + loss_c_2 * conf_pi_2_.view(-1, 1) + loss_c_3 * conf_pi_3_.view(-1, 1) + loss_c_4 * conf_pi_4_.view(-1, 1)
            loss_c = loss_c.view(pos.size()[0], pos.size()[1])
            loss_c[pos] = 0
            loss_c = loss_c.view(num, -1)
            _, loss_idx = loss_c.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_pos = pos.long().sum(1, keepdim=True)
            num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
            neg = idx_rank < num_neg.expand_as(idx_rank)
            pos_idx = pos.unsqueeze(2).expand_as(conf_mu_1)
            neg_idx = neg.unsqueeze(2).expand_as(conf_mu_1)
            batch_conf_1_ = conf_mu_1 + torch.sqrt(conf_var_1) * rand_val_1
            batch_conf_2_ = conf_mu_2 + torch.sqrt(conf_var_2) * rand_val_2
            batch_conf_3_ = conf_mu_3 + torch.sqrt(conf_var_3) * rand_val_3
            batch_conf_4_ = conf_mu_4 + torch.sqrt(conf_var_4) * rand_val_4
            conf_pred_1 = batch_conf_1_[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
            conf_pred_2 = batch_conf_2_[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
            conf_pred_3 = batch_conf_3_[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
            conf_pred_4 = batch_conf_4_[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
            targets_weighted = conf_t[(pos + neg).gt(0)]
            loss_c_1 = log_sum_exp(conf_pred_1) - conf_pred_1.gather(1, targets_weighted.view(-1, 1))
            loss_c_2 = log_sum_exp(conf_pred_2) - conf_pred_2.gather(1, targets_weighted.view(-1, 1))
            loss_c_3 = log_sum_exp(conf_pred_3) - conf_pred_3.gather(1, targets_weighted.view(-1, 1))
            loss_c_4 = log_sum_exp(conf_pred_4) - conf_pred_4.gather(1, targets_weighted.view(-1, 1))
            _conf_pi_1 = conf_pi_1_[(pos + neg).gt(0)]
            _conf_pi_2 = conf_pi_2_[(pos + neg).gt(0)]
            _conf_pi_3 = conf_pi_3_[(pos + neg).gt(0)]
            _conf_pi_4 = conf_pi_4_[(pos + neg).gt(0)]
            loss_c = loss_c_1 * _conf_pi_1.view(-1, 1) + loss_c_2 * _conf_pi_2.view(-1, 1) + loss_c_3 * _conf_pi_3.view(-1, 1) + loss_c_4 * _conf_pi_4.view(-1, 1)
            loss_c = loss_c.sum()
        else:
            conf_pi_1_ = conf_pi_1.view(-1, 1)
            conf_pi_2_ = conf_pi_2.view(-1, 1)
            conf_pi_3_ = conf_pi_3.view(-1, 1)
            conf_pi_4_ = conf_pi_4.view(-1, 1)
            conf_pi_all = torch.stack([conf_pi_1_.reshape(-1), conf_pi_2_.reshape(-1), conf_pi_3_.reshape(-1), conf_pi_4_.reshape(-1)])
            conf_pi_all = conf_pi_all.transpose(0, 1)
            conf_pi_all = torch.softmax(conf_pi_all, dim=1).transpose(0, 1).reshape(-1)
            conf_pi_1_, conf_pi_2_, conf_pi_3_, conf_pi_4_ = torch.split(conf_pi_all, conf_pi_1_.reshape(-1).size(0), dim=0)
            conf_pi_1_ = conf_pi_1_.view(conf_pi_1.size(0), -1)
            conf_pi_2_ = conf_pi_2_.view(conf_pi_2.size(0), -1)
            conf_pi_3_ = conf_pi_3_.view(conf_pi_3.size(0), -1)
            conf_pi_4_ = conf_pi_4_.view(conf_pi_4.size(0), -1)
            conf_var_1 = torch.sigmoid(conf_var_1)
            conf_var_2 = torch.sigmoid(conf_var_2)
            conf_var_3 = torch.sigmoid(conf_var_3)
            conf_var_4 = torch.sigmoid(conf_var_4)
            rand_val_1 = torch.randn(conf_var_1.size(0), conf_var_1.size(1), conf_var_1.size(2))
            rand_val_2 = torch.randn(conf_var_2.size(0), conf_var_2.size(1), conf_var_2.size(2))
            rand_val_3 = torch.randn(conf_var_3.size(0), conf_var_3.size(1), conf_var_3.size(2))
            rand_val_4 = torch.randn(conf_var_4.size(0), conf_var_4.size(1), conf_var_4.size(2))
            batch_conf_1 = (conf_mu_1 + torch.sqrt(conf_var_1) * rand_val_1).view(-1, self.num_classes)
            batch_conf_2 = (conf_mu_2 + torch.sqrt(conf_var_2) * rand_val_2).view(-1, self.num_classes)
            batch_conf_3 = (conf_mu_3 + torch.sqrt(conf_var_3) * rand_val_3).view(-1, self.num_classes)
            batch_conf_4 = (conf_mu_4 + torch.sqrt(conf_var_4) * rand_val_4).view(-1, self.num_classes)
            soft_max = nn.Softmax(dim=1)
            epsi = 10 ** -9
            weighted_softmax_out = soft_max(batch_conf_1) * conf_pi_1_.view(-1, 1) + soft_max(batch_conf_2) * conf_pi_2_.view(-1, 1) + soft_max(batch_conf_3) * conf_pi_3_.view(-1, 1) + soft_max(batch_conf_4) * conf_pi_4_.view(-1, 1)
            softmax_out_log = -torch.log(weighted_softmax_out + epsi)
            loss_c = softmax_out_log.gather(1, conf_t.view(-1, 1))
            loss_c = loss_c.view(pos.size()[0], pos.size()[1])
            loss_c[pos] = 0
            loss_c = loss_c.view(num, -1)
            _, loss_idx = loss_c.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_pos = pos.long().sum(1, keepdim=True)
            num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
            neg = idx_rank < num_neg.expand_as(idx_rank)
            pos_idx = pos.unsqueeze(2).expand_as(conf_mu_1)
            neg_idx = neg.unsqueeze(2).expand_as(conf_mu_1)
            batch_conf_1_ = conf_mu_1 + torch.sqrt(conf_var_1) * rand_val_1
            batch_conf_2_ = conf_mu_2 + torch.sqrt(conf_var_2) * rand_val_2
            batch_conf_3_ = conf_mu_3 + torch.sqrt(conf_var_3) * rand_val_3
            batch_conf_4_ = conf_mu_4 + torch.sqrt(conf_var_4) * rand_val_4
            conf_pred_1 = batch_conf_1_[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
            conf_pred_2 = batch_conf_2_[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
            conf_pred_3 = batch_conf_3_[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
            conf_pred_4 = batch_conf_4_[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
            targets_weighted = conf_t[(pos + neg).gt(0)]
            _conf_pi_1 = conf_pi_1_[(pos + neg).gt(0)]
            _conf_pi_2 = conf_pi_2_[(pos + neg).gt(0)]
            _conf_pi_3 = conf_pi_3_[(pos + neg).gt(0)]
            _conf_pi_4 = conf_pi_4_[(pos + neg).gt(0)]
            weighted_softmax_out = soft_max(conf_pred_1) * _conf_pi_1.view(-1, 1) + soft_max(conf_pred_2) * _conf_pi_2.view(-1, 1) + soft_max(conf_pred_3) * _conf_pi_3.view(-1, 1) + soft_max(conf_pred_4) * _conf_pi_4.view(-1, 1)
            softmax_out_log = -torch.log(weighted_softmax_out + epsi)
            loss_c = softmax_out_log.gather(1, targets_weighted.view(-1, 1))
            loss_c = loss_c.sum()
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c


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


class Detect_GMM(Function):
    """At test time, Detect is the final layer of SSD. Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = [0.1, 0.2]

    def forward(self, prior_data, loc_mu_1=None, loc_var_1=None, loc_pi_1=None, loc_mu_2=None, loc_var_2=None, loc_pi_2=None, loc_mu_3=None, loc_var_3=None, loc_pi_3=None, loc_mu_4=None, loc_var_4=None, loc_pi_4=None, conf_mu_1=None, conf_var_1=None, conf_pi_1=None, conf_mu_2=None, conf_var_2=None, conf_pi_2=None, conf_mu_3=None, conf_var_3=None, conf_pi_3=None, conf_mu_4=None, conf_var_4=None, conf_pi_4=None):
        num = loc_mu_1.size(0)
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 15)
        new_loc = loc_pi_1 * loc_mu_1 + loc_pi_2 * loc_mu_2 + loc_pi_3 * loc_mu_3 + loc_pi_4 * loc_mu_4
        al_uc = loc_pi_1 * loc_var_1 + loc_pi_2 * loc_var_2 + loc_pi_3 * loc_var_3 + loc_pi_4 * loc_var_4
        ep_uc = loc_pi_1 * (loc_mu_1 - new_loc) ** 2 + loc_pi_2 * (loc_mu_2 - new_loc) ** 2 + loc_pi_3 * (loc_mu_3 - new_loc) ** 2 + loc_pi_4 * (loc_mu_4 - new_loc) ** 2
        new_conf = conf_pi_1 * conf_mu_1 + conf_pi_2 * conf_mu_2 + conf_pi_3 * conf_mu_3 + conf_pi_4 * conf_mu_4
        cls_al_uc = conf_pi_1 * conf_var_1 + conf_pi_2 * conf_var_2 + conf_pi_3 * conf_var_3 + conf_pi_4 * conf_var_4
        cls_ep_uc = conf_pi_1 * (conf_mu_1 - new_conf) ** 2 + conf_pi_2 * (conf_mu_2 - new_conf) ** 2 + conf_pi_3 * (conf_mu_3 - new_conf) ** 2 + conf_pi_4 * (conf_mu_4 - new_conf) ** 2
        new_conf = new_conf.view(num, num_priors, self.num_classes).transpose(2, 1)
        cls_al_uc = cls_al_uc.view(num, num_priors, self.num_classes).transpose(2, 1)
        cls_ep_uc = cls_ep_uc.view(num, num_priors, self.num_classes).transpose(2, 1)
        for i in range(num):
            decoded_boxes = decode(new_loc[i], prior_data, self.variance)
            conf_scores = new_conf[i].clone()
            conf_al_clone = cls_al_uc[i].clone()
            conf_ep_clone = cls_ep_uc[i].clone()
            loc_al_uc_clone = al_uc[i].clone()
            loc_ep_uc_clone = ep_uc[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                conf_al = conf_al_clone[cl][c_mask]
                conf_ep = conf_ep_clone[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                loc_al_uc = loc_al_uc_clone[l_mask].view(-1, 4)
                loc_ep_uc = loc_ep_uc_clone[l_mask].view(-1, 4)
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]], loc_al_uc[ids[:count]], loc_ep_uc[ids[:count]], conf_al[ids[:count]].unsqueeze(1), conf_ep[ids[:count]].unsqueeze(1)), 1)
        flt = output.contiguous().view(num, -1, 15)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


coco = {'num_classes': 81, 'lr_steps': (80000, 100000, 120000), 'max_iter': 120000, 'feature_maps': [38, 19, 10, 5, 3, 1], 'min_dim': 300, 'steps': [8, 16, 32, 64, 100, 300], 'min_sizes': [21, 45, 99, 153, 207, 261], 'max_sizes': [45, 99, 153, 207, 261, 315], 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]], 'variance': [0.1, 0.2], 'clip': True, 'name': 'COCO'}


voc300 = {'num_classes': 21, 'lr_steps': (80000, 100000, 120000), 'max_iter': 120000, 'feature_maps': [38, 19, 10, 5, 3, 1], 'min_dim': 300, 'steps': [8, 16, 32, 64, 100, 300], 'min_sizes': [30, 60, 111, 162, 213, 264], 'max_sizes': [60, 111, 162, 213, 264, 315], 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]], 'variance': [0.1, 0.2], 'clip': True, 'name': 'VOC'}


voc512 = {'num_classes': 21, 'lr_steps': (80000, 100000, 120000), 'max_iter': 120000, 'feature_maps': [64, 32, 16, 8, 4, 2, 1], 'min_dim': 512, 'steps': [8, 16, 32, 64, 128, 256, 512], 'min_sizes': [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8], 'max_sizes': [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6], 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]], 'variance': [0.1, 0.2], 'clip': True, 'name': 'VOC'}


class SSD_GMM(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD_GMM, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        if size == 300:
            self.cfg = (coco, voc300)[num_classes == 21]
        else:
            self.cfg = (coco, voc512)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.size = size
        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.loc_mu_1 = nn.ModuleList(head[0])
        self.loc_var_1 = nn.ModuleList(head[1])
        self.loc_pi_1 = nn.ModuleList(head[2])
        self.loc_mu_2 = nn.ModuleList(head[3])
        self.loc_var_2 = nn.ModuleList(head[4])
        self.loc_pi_2 = nn.ModuleList(head[5])
        self.loc_mu_3 = nn.ModuleList(head[6])
        self.loc_var_3 = nn.ModuleList(head[7])
        self.loc_pi_3 = nn.ModuleList(head[8])
        self.loc_mu_4 = nn.ModuleList(head[9])
        self.loc_var_4 = nn.ModuleList(head[10])
        self.loc_pi_4 = nn.ModuleList(head[11])
        self.conf_mu_1 = nn.ModuleList(head[12])
        self.conf_var_1 = nn.ModuleList(head[13])
        self.conf_pi_1 = nn.ModuleList(head[14])
        self.conf_mu_2 = nn.ModuleList(head[15])
        self.conf_var_2 = nn.ModuleList(head[16])
        self.conf_pi_2 = nn.ModuleList(head[17])
        self.conf_mu_3 = nn.ModuleList(head[18])
        self.conf_var_3 = nn.ModuleList(head[19])
        self.conf_pi_3 = nn.ModuleList(head[20])
        self.conf_mu_4 = nn.ModuleList(head[21])
        self.conf_var_4 = nn.ModuleList(head[22])
        self.conf_pi_4 = nn.ModuleList(head[23])
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect_GMM(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected.

            train:
                list of concat outputs from:
                    1: confidence layers
                    2: localization layer
                    3: priorbox layers
        """
        sources = list()
        loc_mu_1 = list()
        loc_var_1 = list()
        loc_pi_1 = list()
        loc_mu_2 = list()
        loc_var_2 = list()
        loc_pi_2 = list()
        loc_mu_3 = list()
        loc_var_3 = list()
        loc_pi_3 = list()
        loc_mu_4 = list()
        loc_var_4 = list()
        loc_pi_4 = list()
        conf_mu_1 = list()
        conf_var_1 = list()
        conf_pi_1 = list()
        conf_mu_2 = list()
        conf_var_2 = list()
        conf_pi_2 = list()
        conf_mu_3 = list()
        conf_var_3 = list()
        conf_pi_3 = list()
        conf_mu_4 = list()
        conf_var_4 = list()
        conf_pi_4 = list()
        for k in range(23):
            x = self.vgg[k](x)
        s = self.L2Norm(x)
        sources.append(s)
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        for x, l_mu_1, l_var_1, l_pi_1, l_mu_2, l_var_2, l_pi_2, l_mu_3, l_var_3, l_pi_3, l_mu_4, l_var_4, l_pi_4, c_mu_1, c_var_1, c_pi_1, c_mu_2, c_var_2, c_pi_2, c_mu_3, c_var_3, c_pi_3, c_mu_4, c_var_4, c_pi_4 in zip(sources, self.loc_mu_1, self.loc_var_1, self.loc_pi_1, self.loc_mu_2, self.loc_var_2, self.loc_pi_2, self.loc_mu_3, self.loc_var_3, self.loc_pi_3, self.loc_mu_4, self.loc_var_4, self.loc_pi_4, self.conf_mu_1, self.conf_var_1, self.conf_pi_1, self.conf_mu_2, self.conf_var_2, self.conf_pi_2, self.conf_mu_3, self.conf_var_3, self.conf_pi_3, self.conf_mu_4, self.conf_var_4, self.conf_pi_4):
            loc_mu_1.append(l_mu_1(x).permute(0, 2, 3, 1).contiguous())
            loc_var_1.append(l_var_1(x).permute(0, 2, 3, 1).contiguous())
            loc_pi_1.append(l_pi_1(x).permute(0, 2, 3, 1).contiguous())
            loc_mu_2.append(l_mu_2(x).permute(0, 2, 3, 1).contiguous())
            loc_var_2.append(l_var_2(x).permute(0, 2, 3, 1).contiguous())
            loc_pi_2.append(l_pi_2(x).permute(0, 2, 3, 1).contiguous())
            loc_mu_3.append(l_mu_3(x).permute(0, 2, 3, 1).contiguous())
            loc_var_3.append(l_var_3(x).permute(0, 2, 3, 1).contiguous())
            loc_pi_3.append(l_pi_3(x).permute(0, 2, 3, 1).contiguous())
            loc_mu_4.append(l_mu_4(x).permute(0, 2, 3, 1).contiguous())
            loc_var_4.append(l_var_4(x).permute(0, 2, 3, 1).contiguous())
            loc_pi_4.append(l_pi_4(x).permute(0, 2, 3, 1).contiguous())
            conf_mu_1.append(c_mu_1(x).permute(0, 2, 3, 1).contiguous())
            conf_var_1.append(c_var_1(x).permute(0, 2, 3, 1).contiguous())
            conf_pi_1.append(c_pi_1(x).permute(0, 2, 3, 1).contiguous())
            conf_mu_2.append(c_mu_2(x).permute(0, 2, 3, 1).contiguous())
            conf_var_2.append(c_var_2(x).permute(0, 2, 3, 1).contiguous())
            conf_pi_2.append(c_pi_2(x).permute(0, 2, 3, 1).contiguous())
            conf_mu_3.append(c_mu_3(x).permute(0, 2, 3, 1).contiguous())
            conf_var_3.append(c_var_3(x).permute(0, 2, 3, 1).contiguous())
            conf_pi_3.append(c_pi_3(x).permute(0, 2, 3, 1).contiguous())
            conf_mu_4.append(c_mu_4(x).permute(0, 2, 3, 1).contiguous())
            conf_var_4.append(c_var_4(x).permute(0, 2, 3, 1).contiguous())
            conf_pi_4.append(c_pi_4(x).permute(0, 2, 3, 1).contiguous())
        loc_mu_1 = torch.cat([o.view(o.size(0), -1) for o in loc_mu_1], 1)
        loc_var_1 = torch.cat([o.view(o.size(0), -1) for o in loc_var_1], 1)
        loc_pi_1 = torch.cat([o.view(o.size(0), -1) for o in loc_pi_1], 1)
        loc_mu_2 = torch.cat([o.view(o.size(0), -1) for o in loc_mu_2], 1)
        loc_var_2 = torch.cat([o.view(o.size(0), -1) for o in loc_var_2], 1)
        loc_pi_2 = torch.cat([o.view(o.size(0), -1) for o in loc_pi_2], 1)
        loc_mu_3 = torch.cat([o.view(o.size(0), -1) for o in loc_mu_3], 1)
        loc_var_3 = torch.cat([o.view(o.size(0), -1) for o in loc_var_3], 1)
        loc_pi_3 = torch.cat([o.view(o.size(0), -1) for o in loc_pi_3], 1)
        loc_mu_4 = torch.cat([o.view(o.size(0), -1) for o in loc_mu_4], 1)
        loc_var_4 = torch.cat([o.view(o.size(0), -1) for o in loc_var_4], 1)
        loc_pi_4 = torch.cat([o.view(o.size(0), -1) for o in loc_pi_4], 1)
        conf_mu_1 = torch.cat([o.view(o.size(0), -1) for o in conf_mu_1], 1)
        conf_var_1 = torch.cat([o.view(o.size(0), -1) for o in conf_var_1], 1)
        conf_pi_1 = torch.cat([o.view(o.size(0), -1) for o in conf_pi_1], 1)
        conf_mu_2 = torch.cat([o.view(o.size(0), -1) for o in conf_mu_2], 1)
        conf_var_2 = torch.cat([o.view(o.size(0), -1) for o in conf_var_2], 1)
        conf_pi_2 = torch.cat([o.view(o.size(0), -1) for o in conf_pi_2], 1)
        conf_mu_3 = torch.cat([o.view(o.size(0), -1) for o in conf_mu_3], 1)
        conf_var_3 = torch.cat([o.view(o.size(0), -1) for o in conf_var_3], 1)
        conf_pi_3 = torch.cat([o.view(o.size(0), -1) for o in conf_pi_3], 1)
        conf_mu_4 = torch.cat([o.view(o.size(0), -1) for o in conf_mu_4], 1)
        conf_var_4 = torch.cat([o.view(o.size(0), -1) for o in conf_var_4], 1)
        conf_pi_4 = torch.cat([o.view(o.size(0), -1) for o in conf_pi_4], 1)
        if self.phase == 'test':
            loc_var_1 = torch.sigmoid(loc_var_1)
            loc_var_2 = torch.sigmoid(loc_var_2)
            loc_var_3 = torch.sigmoid(loc_var_3)
            loc_var_4 = torch.sigmoid(loc_var_4)
            loc_pi_1 = loc_pi_1.view(-1, 4)
            loc_pi_2 = loc_pi_2.view(-1, 4)
            loc_pi_3 = loc_pi_3.view(-1, 4)
            loc_pi_4 = loc_pi_4.view(-1, 4)
            pi_all = torch.stack([loc_pi_1.reshape(-1), loc_pi_2.reshape(-1), loc_pi_3.reshape(-1), loc_pi_4.reshape(-1)])
            pi_all = pi_all.transpose(0, 1)
            pi_all = torch.softmax(pi_all, dim=1).transpose(0, 1).reshape(-1)
            loc_pi_1, loc_pi_2, loc_pi_3, loc_pi_4 = torch.split(pi_all, loc_pi_1.reshape(-1).size(0), dim=0)
            loc_pi_1 = loc_pi_1.view(-1, 4)
            loc_pi_2 = loc_pi_2.view(-1, 4)
            loc_pi_3 = loc_pi_3.view(-1, 4)
            loc_pi_4 = loc_pi_4.view(-1, 4)
            conf_var_1 = torch.sigmoid(conf_var_1)
            conf_var_2 = torch.sigmoid(conf_var_2)
            conf_var_3 = torch.sigmoid(conf_var_3)
            conf_var_4 = torch.sigmoid(conf_var_4)
            conf_pi_1 = conf_pi_1.view(-1, 1)
            conf_pi_2 = conf_pi_2.view(-1, 1)
            conf_pi_3 = conf_pi_3.view(-1, 1)
            conf_pi_4 = conf_pi_4.view(-1, 1)
            conf_pi_all = torch.stack([conf_pi_1.reshape(-1), conf_pi_2.reshape(-1), conf_pi_3.reshape(-1), conf_pi_4.reshape(-1)])
            conf_pi_all = conf_pi_all.transpose(0, 1)
            conf_pi_all = torch.softmax(conf_pi_all, dim=1).transpose(0, 1).reshape(-1)
            conf_pi_1, conf_pi_2, conf_pi_3, conf_pi_4 = torch.split(conf_pi_all, conf_pi_1.reshape(-1).size(0), dim=0)
            conf_pi_1 = conf_pi_1.view(-1, 1)
            conf_pi_2 = conf_pi_2.view(-1, 1)
            conf_pi_3 = conf_pi_3.view(-1, 1)
            conf_pi_4 = conf_pi_4.view(-1, 1)
            output = self.detect(self.priors.type(type(x.data)), loc_mu_1.view(loc_mu_1.size(0), -1, 4), loc_var_1.view(loc_var_1.size(0), -1, 4), loc_pi_1.view(loc_var_1.size(0), -1, 4), loc_mu_2.view(loc_mu_2.size(0), -1, 4), loc_var_2.view(loc_var_2.size(0), -1, 4), loc_pi_2.view(loc_var_2.size(0), -1, 4), loc_mu_3.view(loc_mu_3.size(0), -1, 4), loc_var_3.view(loc_var_3.size(0), -1, 4), loc_pi_3.view(loc_var_3.size(0), -1, 4), loc_mu_4.view(loc_mu_4.size(0), -1, 4), loc_var_4.view(loc_var_4.size(0), -1, 4), loc_pi_4.view(loc_var_4.size(0), -1, 4), self.softmax(conf_mu_1.view(conf_mu_1.size(0), -1, self.num_classes)), conf_var_1.view(conf_var_1.size(0), -1, self.num_classes), conf_pi_1.view(conf_var_1.size(0), -1, 1), self.softmax(conf_mu_2.view(conf_mu_2.size(0), -1, self.num_classes)), conf_var_2.view(conf_var_2.size(0), -1, self.num_classes), conf_pi_2.view(conf_var_2.size(0), -1, 1), self.softmax(conf_mu_3.view(conf_mu_3.size(0), -1, self.num_classes)), conf_var_3.view(conf_var_3.size(0), -1, self.num_classes), conf_pi_3.view(conf_var_3.size(0), -1, 1), self.softmax(conf_mu_4.view(conf_mu_4.size(0), -1, self.num_classes)), conf_var_4.view(conf_var_4.size(0), -1, self.num_classes), conf_pi_4.view(conf_var_4.size(0), -1, 1))
        else:
            output = self.priors, loc_mu_1.view(loc_mu_1.size(0), -1, 4), loc_var_1.view(loc_var_1.size(0), -1, 4), loc_pi_1.view(loc_pi_1.size(0), -1, 4), loc_mu_2.view(loc_mu_2.size(0), -1, 4), loc_var_2.view(loc_var_2.size(0), -1, 4), loc_pi_2.view(loc_pi_2.size(0), -1, 4), loc_mu_3.view(loc_mu_3.size(0), -1, 4), loc_var_3.view(loc_var_3.size(0), -1, 4), loc_pi_3.view(loc_pi_3.size(0), -1, 4), loc_mu_4.view(loc_mu_4.size(0), -1, 4), loc_var_4.view(loc_var_4.size(0), -1, 4), loc_pi_4.view(loc_pi_4.size(0), -1, 4), conf_mu_1.view(conf_mu_1.size(0), -1, self.num_classes), conf_var_1.view(conf_var_1.size(0), -1, self.num_classes), conf_pi_1.view(conf_pi_1.size(0), -1, 1), conf_mu_2.view(conf_mu_2.size(0), -1, self.num_classes), conf_var_2.view(conf_var_2.size(0), -1, self.num_classes), conf_pi_2.view(conf_pi_2.size(0), -1, 1), conf_mu_3.view(conf_mu_3.size(0), -1, self.num_classes), conf_var_3.view(conf_var_3.size(0), -1, self.num_classes), conf_pi_3.view(conf_pi_3.size(0), -1, 1), conf_mu_4.view(conf_mu_4.size(0), -1, self.num_classes), conf_var_4.view(conf_var_4.size(0), -1, self.num_classes), conf_pi_4.view(conf_pi_4.size(0), -1, 1)
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            None
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            None
        else:
            None


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

class Test_NVlabs_AL_MDN(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

