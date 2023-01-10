import sys
_module = sys.modules[__name__]
del sys
data = _module
config = _module
hand_dataset = _module
create_trainval = _module
eval = _module
eval_speed = _module
layers = _module
box_utils = _module
build = _module
functions = _module
detection = _module
prior_box = _module
modules = _module
l2norm = _module
multibox_loss = _module
models = _module
mobilenet_utils = _module
ssd_mobilenet = _module
ssd_new_mobilenet = _module
ssd_new_mobilenet_FFA = _module
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


import numpy as np


import random


import torch.utils.data as data


import torch.nn as nn


import torch.backends.cudnn as cudnn


import torchvision.transforms as transforms


from torch.autograd import Variable


import time


from scipy.io import loadmat


from torch.autograd import Function


from math import sqrt as sqrt


from itertools import product as product


import torch.nn.init as init


import torch.nn.functional as F


from collections import namedtuple


from collections import OrderedDict


from collections import Iterable


import torch.optim as optim


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
        x /= norm
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


def encode_v2(matched, priors, variances):
    g_cxcy = (matched[:, :2] + matched[:, 2:4]) / 2 - priors[:, :2]
    g_cxcy /= variances[0] * priors[:, 2:]
    g_wh = (matched[:, 2:4] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    g_wxy = torch.log(matched[:, 4:6] / priors[:, 2:] + 0.1) / variances[1]
    g_hxy = torch.log(matched[:, 6:8] / priors[:, 2:] + 0.1) / variances[1]
    wrist_vec = (matched[:, 8:10] - priors[:, :2]) / (variances[0] * priors[:, 2:])
    return torch.cat([g_cxcy, g_wh, g_wxy, g_hxy, wrist_vec], 1)


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
        A ^ B / A || B = A ^ B / (area(A) + area(B) - A ^ B)
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


def match_v2(threshold, gt_boxes, truths, priors, variances, labels, loc_t, conf_t, idx):
    overlaps = jaccard(gt_boxes, point_form(priors))
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
    loc = encode_v2(matches, priors, variances)
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
        L(x,c,l,g) = (Lconf(x, c) + aLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by a which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
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
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = priors.size(0)
        num_classes = self.num_classes
        loc_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            gt_boxes = targets[idx][:, 0:4].data
            match_v2(self.threshold, gt_boxes, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t
            conf_t = conf_t
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        pos = conf_t > 0
        num_pos = pos.sum(1, keepdim=True)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 10)
        loc_t = loc_t[pos_idx].view(-1, 10)
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
        loss_l = loss_l / N
        loss_c = loss_c / N
        return loss_l, loss_c


class Conv2d_tf(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get('padding', 'SAME')
        kwargs['padding'] = 0
        if not isinstance(self.stride, Iterable):
            self.stride = self.stride, self.stride
        if not isinstance(self.dilation, Iterable):
            self.dilation = self.dilation, self.dilation

    def forward(self, input):
        if self.padding == 'VALID':
            return F.conv2d(input, self.weight, self.bias, self.stride, padding=0, dilation=self.dilation, groups=self.groups)
        input_rows = input.size(2)
        filter_rows = self.weight.size(2)
        effective_filter_size_rows = (filter_rows - 1) * self.dilation[0] + 1
        out_rows = (input_rows + self.stride[0] - 1) // self.stride[0]
        padding_rows = max(0, (out_rows - 1) * self.stride[0] + effective_filter_size_rows - input_rows)
        rows_odd = padding_rows % 2 != 0
        input_cols = input.size(3)
        filter_cols = self.weight.size(3)
        effective_filter_size_cols = (filter_cols - 1) * self.dilation[1] + 1
        out_cols = (input_cols + self.stride[1] - 1) // self.stride[1]
        padding_cols = max(0, (out_cols - 1) * self.stride[1] + effective_filter_size_cols - input_cols)
        cols_odd = padding_cols % 2 != 0
        if rows_odd or cols_odd:
            input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])
        return F.conv2d(input, self.weight, self.bias, self.stride, padding=(padding_rows // 2, padding_cols // 2), dilation=self.dilation, groups=self.groups)


def conv_dw(in_channels, kernel_size=3, stride=1, padding='SAME', dilation=1):
    return nn.Sequential(Conv2d_tf(in_channels, in_channels, kernel_size, stride, padding=padding, groups=in_channels, dilation=dilation, bias=False), nn.BatchNorm2d(in_channels, eps=0.001), nn.ReLU6(inplace=True))


def conv_pw(in_channels, out_channels, kernel_size=1, stride=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, bias=False), nn.BatchNorm2d(out_channels, eps=0.001), nn.ReLU6(inplace=True))


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def expand_input(n, divisible_by=8):
    return lambda num_inputs, **_: _make_divisible(num_inputs * n, divisible_by)


def make_fixed_padding(kernel_size, rate=1):
    """Pads the input along the spatial dimensions independently of input size.

    Pads the input such that if it was used in a convolution with 'VALID' padding,
    the output would have the same dimensions as if the unpadded input was used
    in a convolution with 'SAME' padding.

    Args:
        kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
        rate: An integer, rate for atrous convolution.

    Returns:
        output: A padding module.
    """
    if not isinstance(kernel_size, Iterable):
        kernel_size = kernel_size, kernel_size
    kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1), kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
    pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
    pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
    pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
    padding_module = nn.ZeroPad2d((pad_beg[0], pad_end[0], pad_beg[1], pad_end[1]))
    return padding_module


class ExpandedConv(nn.Module):

    def __init__(self, in_channels, out_channels, expansion_size=expand_input(6), kernel_size=3, stride=1, layer_rate=1, residual=True, use_explicit_padding=False, **unused_kwargs):
        super(ExpandedConv, self).__init__()
        self.residual = residual and stride == 1 and in_channels == out_channels
        inner_size = expansion_size(in_channels)
        tmp = OrderedDict()
        if inner_size > in_channels:
            tmp['expand'] = conv_pw(in_channels, inner_size, 1, stride=1)
        if use_explicit_padding:
            tmp.update({'Pad': make_fixed_padding(kernel_size, layer_rate)})
            padding = 'VALID'
        else:
            padding = 'SAME'
        tmp['depthwise'] = conv_dw(inner_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=layer_rate)
        tmp['project'] = nn.Sequential(nn.Conv2d(inner_size, out_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(out_channels))
        self.module = nn.Sequential(tmp)

    def forward(self, x):
        if self.residual:
            return x + self.module(x)
        else:
            return self.module(x)


def decode_v2(loc, priors, variances):
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * torch.exp(loc[:, 2:4] * variances[1]), torch.clamp(priors[:, 2:] * (torch.exp(loc[:, 4:6] * variances[1]) - 0.1), min=0), torch.clamp(priors[:, 2:] * (torch.exp(loc[:, 6:8] * variances[1]) - 0.1), min=0), priors[:, :2] + loc[:, 8:10] * variances[0] * priors[:, 2:]), 1)
    boxes[:, :2] -= boxes[:, 2:4] / 2
    boxes[:, 2:4] += boxes[:, :2]
    return boxes


def nms_new(boxes, scores, overlap=0.5, top_k=200):
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
    dets = torch.cat((boxes, scores), dim=1)
    dets = dets.type(torch.FloatTensor)
    if boxes.numel() == 0:
        return keep
    scores = dets[:, 4]
    idx = scores.sort(0, descending=True)[1]
    idx = idx[0:top_k]
    dets = dets[idx].contiguous()
    scores2 = dets[:, 4]
    idx2 = scores2.sort(0, descending=True)[1]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = torch.mul(x2 - x1, y2 - y1)
    keep = torch.LongTensor(dets.size(0))
    num_out = torch.LongTensor(1)
    nms.cpu_nms(keep, num_out, dets, idx2, areas, overlap)
    keep = idx[keep[:num_out[0]]].contiguous()
    return keep, num_out[0]


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
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
        self.variance = cfg['variance']
        self.output = torch.zeros(1, self.num_classes, self.top_k, 11)

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
        self.output.zero_()
        if num == 1:
            conf_preds = conf_data.t().contiguous().unsqueeze(0)
        else:
            conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
            self.output.expand_(num, self.num_classes, self.top_k, 11)
        for i in range(num):
            decoded_boxes = decode_v2(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()
            num_det = 0
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 10)
                ids, count = nms_new(boxes[:, :4], scores.view(-1, 1), self.nms_thresh, self.top_k)
                self.output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
        flt = self.output.view(-1, 11)
        _, idx = flt[:, 0].sort(0)
        _, rank = idx.sort(0)
        flt[(rank >= self.top_k).unsqueeze(1).expand_as(flt)].fill_(0)
        return self.output


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

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
        if self.version == 'v2':
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
        elif self.version == 'v3' or self.version == 'v2_noclip':
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
        else:
            for i, k in enumerate(self.feature_maps):
                step_x = step_y = self.image_size / k
                for h, w in product(range(k), repeat=2):
                    c_x = (w + 0.5) * step_x
                    c_y = (h + 0.5) * step_y
                    c_w = c_h = self.min_sizes[i] / 2
                    s_k = self.image_size
                    mean += [(c_x - c_w) / s_k, (c_y - c_h) / s_k, (c_x + c_w) / s_k, (c_y + c_h) / s_k]
                    if self.max_sizes[i] > 0:
                        c_w = c_h = sqrt(self.min_sizes[i] * self.max_sizes[i]) / 2
                        mean += [(c_x - c_w) / s_k, (c_y - c_h) / s_k, (c_x + c_w) / s_k, (c_y + c_h) / s_k]
                    for ar in self.aspect_ratios[i]:
                        if not abs(ar - 1) < 1e-06:
                            if ar != 0:
                                c_w = self.min_sizes[i] * sqrt(ar) / 2.0
                                c_h = self.min_sizes[i] / sqrt(ar) / 2.0
                                mean += [(c_x - c_w) / s_k, (c_y - c_h) / s_k, (c_x + c_w) / s_k, (c_y + c_h) / s_k]
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def extra_layers():
    layers = OrderedDict()

    def conv_module1(in_channels, inter_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True), nn.ReLU6(inplace=True), nn.Conv2d(inter_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True), nn.ReLU6(inplace=True))

    def conv_module2(in_channels, inter_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True), nn.ReLU6(inplace=True), nn.Conv2d(inter_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True), nn.ReLU6(inplace=True))
    layers['Conv2d_14'] = conv_module1(1024, 256, 512)
    layers['Conv2d_15'] = conv_module2(512, 128, 256)
    layers['Conv2d_16'] = conv_module2(256, 128, 256)
    return nn.Sequential(layers)


mbox = {'300': [4, 6, 6, 6, 4, 4], '512': []}


Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])


DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])


_CONV_DEFS = [Conv(kernel=[3, 3], stride=2, depth=32), DepthSepConv(kernel=[3, 3], stride=1, depth=64), DepthSepConv(kernel=[3, 3], stride=2, depth=128), DepthSepConv(kernel=[3, 3], stride=1, depth=128), DepthSepConv(kernel=[3, 3], stride=2, depth=256), DepthSepConv(kernel=[3, 3], stride=1, depth=256), DepthSepConv(kernel=[3, 3], stride=1, depth=512), DepthSepConv(kernel=[3, 3], stride=1, depth=512), DepthSepConv(kernel=[3, 3], stride=1, depth=512), DepthSepConv(kernel=[3, 3], stride=2, depth=512), DepthSepConv(kernel=[3, 3], stride=1, depth=512), DepthSepConv(kernel=[3, 3], stride=1, depth=512), DepthSepConv(kernel=[3, 3], stride=2, depth=1024), DepthSepConv(kernel=[3, 3], stride=1, depth=1024)]


def mobilenet_v1_base(final_endpoint='features.Conv2d_13_pointwise', min_depth=8, depth_multiplier=1.0, conv_defs=_CONV_DEFS, output_stride=None):
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = OrderedDict()
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    if conv_defs is None:
        conv_defs = _CONV_DEFS
    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')

    def conv_bn(in_channels, out_channels, kernel_size=3, stride=1):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1, bias=False), nn.BatchNorm2d(out_channels, eps=0.001), nn.ReLU6(inplace=True))

    def conv_dw(in_channels, kernel_size=3, stride=1, dilation=1):
        return nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, stride, 1, groups=in_channels, dilation=dilation, bias=False), nn.BatchNorm2d(in_channels, eps=0.001), nn.ReLU6(inplace=True))

    def conv_pw(in_channels, out_channels, kernel_size=1, stride=1, dilation=1):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, bias=False), nn.BatchNorm2d(out_channels, eps=0.001), nn.ReLU6(inplace=True))
    current_stride = 1
    rate = 1
    in_channels = 3
    for i, conv_def in enumerate(conv_defs):
        end_point_base = 'features.Conv2d_%d' % i
        if output_stride is not None and current_stride == output_stride:
            layer_stride = 1
            layer_rate = rate
            rate *= conv_def.stride
        else:
            layer_stride = conv_def.stride
            layer_rate = 1
            current_stride *= conv_def.stride
        out_channels = depth(conv_def.depth)
        if isinstance(conv_def, Conv):
            end_point = end_point_base
            end_points[end_point] = conv_bn(in_channels, out_channels, conv_def.kernel, stride=conv_def.stride)
            if end_point == final_endpoint:
                return nn.Sequential(end_points)
        elif isinstance(conv_def, DepthSepConv):
            end_points[end_point_base] = nn.Sequential(OrderedDict([('depthwise', conv_dw(in_channels, conv_def.kernel, stride=layer_stride, dilation=layer_rate)), ('pointwise', conv_pw(in_channels, out_channels, 1, stride=1))]))
            if end_point_base + '_pointwise' == final_endpoint:
                return nn.Sequential(end_points)
        else:
            raise ValueError('Unknown convolution type %s for layer %d' % (conv_def.ltype, i))
        in_channels = out_channels
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


v2 = {'feature_maps': [38, 19, 10, 5, 3, 1], 'min_dim': 300, 'steps': [8, 16, 32, 64, 100, 300], 'min_sizes': [30, 60, 111, 162, 213, 264], 'max_sizes': [60, 111, 162, 213, 264, 315], 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]], 'variance': [0.1, 0.2], 'clip': True, 'name': 'v2'}


class SSD(nn.Module):
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
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.priorbox = PriorBox(v2)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = 300
        self.src_names = ['features.Conv2d_8', 'features.Conv2d_11', 'features.Conv2d_13', 'Conv2d_14', 'Conv2d_15', 'Conv2d_16']
        self.src_num = len(self.src_names)
        self.src_channels = [512, 512, 1024, 512, 256, 256]
        self.feat_channels = [256, 256, 256, 256, 256, 256]
        self.mobilenet = mobilenet_v1_base()
        self.extras = extra_layers()
        latlayer = list()
        for i in range(self.src_num):
            in_channel = self.src_channels[i]
            latlayer += [nn.Conv2d(in_channel, 256, kernel_size=1, stride=1, padding=0)]
        self.latlayer = nn.ModuleList(latlayer)
        toplayer = list()
        for i in range(self.src_num):
            if i >= self.src_num - 2:
                toplayer += [None]
            else:
                toplayer += [nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, groups=256, bias=False), nn.BatchNorm2d(256, eps=0.001), nn.Conv2d(256, 256, 1, 1, 0, bias=False), nn.BatchNorm2d(256, eps=0.001))]
        self.toplayer = nn.ModuleList(toplayer)
        head = self.multibox(mbox['300'], num_classes)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

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
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        all_modules = self.mobilenet._modules.copy()
        all_modules.update(self.extras._modules)
        all_modules = all_modules.items()
        for name, module in all_modules:
            x = module(x)
            if name in self.src_names:
                sources.append(x)
        features = [None for i in range(self.src_num)]
        for i in range(self.src_num - 1, -1, -1):
            if i >= self.src_num - 2:
                features[i] = self.latlayer[i](sources[i])
            else:
                features[i] = self.toplayer[i](self.upsample_add(features[i + 1], self.latlayer[i](sources[i])))
        for x, l, c in zip(features, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == 'test':
            output = self.detect(loc.view(loc.size(0), -1, 10), self.softmax(conf.view(-1, self.num_classes)), self.priors.type(type(x.data)))
        else:
            output = loc.view(loc.size(0), -1, 10), conf.view(conf.size(0), -1, self.num_classes), self.priors
        return output

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        x = F.upsample(x, scale_factor=2, mode='nearest')
        return x[:, :, :H, :W] + y

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            None
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            None
        else:
            None

    def multibox(self, cfg, num_classes):
        loc_layers = []
        conf_layers = []
        for k, in_channels in enumerate(self.feat_channels):
            loc_layers += [nn.Conv2d(in_channels, cfg[k] * 10, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(in_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
        return loc_layers, conf_layers


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv2d_tf,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L2Norm,
     lambda: ([], {'n_channels': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_yangli18_hand_detection(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

