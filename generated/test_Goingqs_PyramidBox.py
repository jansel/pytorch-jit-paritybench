import sys
_module = sys.modules[__name__]
del sys
data = _module
config = _module
widerface = _module
layers = _module
box_utils = _module
functions = _module
detection = _module
prior_box = _module
modules = _module
l2norm = _module
multibox_loss = _module
pyramid = _module
test = _module
train = _module
utils = _module
augmentations = _module

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


import torch


import torch.nn as nn


from torch.autograd import Function


from torch.autograd import Variable


import torch.nn.init as init


import torch.nn.functional as F


import torch.backends.cudnn as cudnn


import scipy.io as sio


import numpy as np


import math


import torch.optim as optim


import torch.utils.data as data


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
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x
            ) * x
        return out


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes
        [:, 2:] / 2), 1)


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
    if A * B * 2 / 1024 / 1024 * 4 > 1000:
        print('Warning! Memory is:', A * B * 2 / 1024 / 1024 * 4, 'MB')
        box_a_cpu = box_a.cpu()
        box_b_cpu = box_b.cpu()
        max_xy_cpu = torch.min(box_a_cpu[:, 2:].unsqueeze(1).expand(A, B, 2
            ), box_b_cpu[:, 2:].unsqueeze(0).expand(A, B, 2))
        max_xy_cpu = torch.max(box_a_cpu[:, :2].unsqueeze(1).expand(A, B, 2
            ), box_b_cpu[:, :2].unsqueeze(0).expand(A, B, 2))
        max_xy_cpu -= max_xy_cpu
        max_xy_cpu.clamp_(min=0)
        res_cpu = max_xy_cpu[:, :, (0)] * max_xy_cpu[:, :, (1)]
        res = res_cpu
    else:
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b
            [:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b
            [:, :2].unsqueeze(0).expand(A, B, 2))
        max_xy -= min_xy
        max_xy.clamp_(min=0)
        res = max_xy[:, :, (0)] * max_xy[:, :, (1)]
    return res


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
    if not inter.is_cuda:
        box_a_cpu = box_a.cpu()
        box_b_cpu = box_b.cpu()
        area_a_cpu = ((box_a_cpu[:, (2)] - box_a_cpu[:, (0)]) * (box_a_cpu[
            :, (3)] - box_a_cpu[:, (1)])).unsqueeze(1).expand_as(inter)
        area_b_cpu = ((box_b_cpu[:, (2)] - box_b_cpu[:, (0)]) * (box_b_cpu[
            :, (3)] - box_b_cpu[:, (1)])).unsqueeze(0).expand_as(inter)
        union_cpu = area_a_cpu + area_b_cpu - inter.cpu()
        return inter / union_cpu
    else:
        area_a = ((box_a[:, (2)] - box_a[:, (0)]) * (box_a[:, (3)] - box_a[
            :, (1)])).unsqueeze(1).expand_as(inter)
        area_b = ((box_b[:, (2)] - box_b[:, (0)]) * (box_b[:, (3)] - box_b[
            :, (1)])).unsqueeze(0).expand_as(inter)
        union = area_a + area_b - inter
        return inter / union


def matchNoBipartite(threshold, truths, priors, variances, labels, loc_t,
    conf_t, idx):
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
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    if not best_truth_overlap.is_cuda:
        best_truth_overlap = best_truth_overlap.cuda()
        best_truth_idx = best_truth_idx.cuda()
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx] + 1
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf


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
    if not best_truth_overlap.is_cuda:
        best_prior_overlap = best_prior_overlap.cuda()
        best_prior_idx = best_prior_idx.cuda()
        best_truth_overlap = best_truth_overlap.cuda()
        best_truth_idx = best_truth_idx.cuda()
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


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


_global_config['variance'] = 4


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

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
        bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
        bipartite=True, use_gpu=True):
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
        self.bipartite = bipartite
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
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, (-1)].data
            defaults = priors.data
            if self.bipartite:
                match(self.threshold, truths, defaults, self.variance,
                    labels, loc_t, conf_t, idx)
            else:
                matchNoBipartite(self.threshold, truths, defaults, self.
                    variance, labels, loc_t, conf_t, idx)
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
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view
            (-1, 1))
        loss_c[pos] = 0
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes
            )
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        N = num_pos.data.sum()
        if N == 0:
            N = num
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c


class ConvBN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
        padding=0, relu=False):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=
            kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class SSHContext(nn.Module):

    def __init__(self, channels, Xchannels=256):
        super(SSHContext, self).__init__()
        self.conv1 = nn.Conv2d(channels, Xchannels, kernel_size=3, stride=1,
            padding=1)
        self.conv2 = nn.Conv2d(channels, Xchannels // 2, kernel_size=3,
            dilation=2, stride=1, padding=2)
        self.conv2_1 = nn.Conv2d(Xchannels // 2, Xchannels // 2,
            kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(Xchannels // 2, Xchannels // 2,
            kernel_size=3, dilation=2, stride=1, padding=2)
        self.conv2_2_1 = nn.Conv2d(Xchannels // 2, Xchannels // 2,
            kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x), inplace=True)
        x2_1 = F.relu(self.conv2_1(x2), inplace=True)
        x2_2 = F.relu(self.conv2_2(x2), inplace=True)
        x2_2 = F.relu(self.conv2_2_1(x2_2), inplace=True)
        return torch.cat([x1, x2_1, x2_2], 1)


class ContextTexture(nn.Module):
    """docstring for ContextTexture """

    def __init__(self, **channels):
        super(ContextTexture, self).__init__()
        self.up_conv = nn.Conv2d(channels['up'], channels['main'],
            kernel_size=1)
        self.main_conv = nn.Conv2d(channels['main'], channels['main'],
            kernel_size=1)

    def forward(self, up, main):
        up = self.up_conv(up)
        main = self.main_conv(main)
        _, _, H, W = main.size()
        res = F.upsample(up, scale_factor=2, mode='bilinear')
        if res.size(2) != main.size(2) or res.size(3) != main.size(3):
            res = res[:, :, 0:H, 0:W]
        res = res + main
        return res


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size
            =1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, self.
                expansion * planes, kernel_size=1, stride=stride, bias=
                False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out, inplace=True)
        return out


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
    x1 = boxes[:, (0)]
    y1 = boxes[:, (1)]
    x2 = boxes[:, (2)]
    y2 = boxes[:, (3)]
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
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:,
        2:], priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


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
        self.nms_top_k = 5000

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
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, self.num_classes
            ).transpose(2, 1)
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes, scores, self.nms_thresh, min(boxes.
                    shape[0], self.nms_top_k))
                select_count = min(count, self.top_k)
                output[(i), (cl), :select_count] = torch.cat((scores[ids[:
                    select_count]].unsqueeze(1), boxes[ids[:select_count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, (0)].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


class PriorBoxLayer(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """

    def __init__(self, width, height, stride=[4, 8, 16, 32, 64, 128], box=[
        16, 32, 64, 128, 256, 512], scale=[1, 1, 1, 1, 1, 1], aspect_ratios
        =[[], [], [], [], [], []]):
        super(PriorBoxLayer, self).__init__()
        self.width = width
        self.height = height
        self.stride = stride
        self.box = box
        self.scales = scale
        self.aspect_ratios = aspect_ratios

    def forward(self, prior_idx, f_width, f_height):
        mean = []
        for i in range(f_height):
            for j in range(f_width):
                for scale in range(self.scales[prior_idx]):
                    box_scale = (2 ** (1 / 3)) ** scale
                    cx = (j + 0.5) * self.stride[prior_idx] / self.width
                    cy = (i + 0.5) * self.stride[prior_idx] / self.height
                    side_x = self.box[prior_idx] * box_scale / self.width
                    side_y = self.box[prior_idx] * box_scale / self.height
                    mean += [cx, cy, side_x, side_y]
                    for ar in self.aspect_ratios[prior_idx]:
                        mean += [cx, cy, side_x / sqrt(ar), side_y * sqrt(ar)]
        output = torch.Tensor(mean).view(-1, 4)
        return output


class SFD(nn.Module):

    def __init__(self, block, num_blocks, phase, num_classes, size):
        super(SFD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.priorbox = PriorBoxLayer(size, size, stride=[4, 8, 16, 32, 64,
            128])
        self.priors = None
        self.priorbox_head = PriorBoxLayer(size, size, stride=[8, 16, 32, 
            64, 128, 128])
        self.priors_head = None
        self.priorbox_body = PriorBoxLayer(size, size, stride=[16, 32, 64, 
            128, 128, 128])
        self.priors_body = None
        self.size = size
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer5 = nn.Sequential(*[nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 512,
            kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(512), nn.
            ReLU(inplace=True)])
        self.layer6 = nn.Sequential(*[nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 256,
            kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(256), nn.
            ReLU(inplace=True)])
        self.conv3_ct_py = ContextTexture(up=512, main=256)
        self.conv4_ct_py = ContextTexture(up=1024, main=512)
        self.conv5_ct_py = ContextTexture(up=2048, main=1024)
        self.latlayer_fc = nn.Conv2d(2048, 2048, kernel_size=1)
        self.latlayer_c6 = nn.Conv2d(512, 512, kernel_size=1)
        self.latlayer_c7 = nn.Conv2d(256, 256, kernel_size=1)
        self.smooth_c3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_c4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.smooth_c5 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.conv2_SSH = SSHContext(256, 256)
        self.conv3_SSH = SSHContext(512, 256)
        self.conv4_SSH = SSHContext(1024, 256)
        self.conv5_SSH = SSHContext(2048, 256)
        self.conv6_SSH = SSHContext(512, 256)
        self.conv7_SSH = SSHContext(256, 256)
        self.SSHchannels = [512, 512, 512, 512, 512, 512]
        loc = []
        conf = []
        for i in range(6):
            loc.append(nn.Conv2d(self.SSHchannels[i], 4, kernel_size=3,
                stride=1, padding=1))
            conf.append(nn.Conv2d(self.SSHchannels[i], 4, kernel_size=3,
                stride=1, padding=1))
        self.face_loc = nn.ModuleList(loc)
        self.face_conf = nn.ModuleList(conf)
        head_loc = []
        head_conf = []
        for i in range(5):
            head_loc.append(nn.Conv2d(self.SSHchannels[i + 1], 4,
                kernel_size=3, stride=1, padding=1))
            head_conf.append(nn.Conv2d(self.SSHchannels[i + 1], 2,
                kernel_size=3, stride=1, padding=1))
        self.head_loc = nn.ModuleList(head_loc)
        self.head_conf = nn.ModuleList(head_conf)
        """body_loc = []
        body_conf = []
        for i in range(4):
            body_loc.append(nn.Conv2d(self.SSHchannels[i+2],4,kernel_size=3,stride=1,padding=1))
            body_conf.append(nn.Conv2d(self.SSHchannels[i+2],2,kernel_size=3,stride=1,padding=1))

        self.body_loc = nn.ModuleList(body_loc)
        self.body_conf = nn.ModuleList(body_conf)"""
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 750, 0.05, 0.3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()
        head_loc = list()
        head_conf = list()
        body_conf = list()
        body_loc = list()
        c1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        c6 = self.layer5(c5)
        c7 = self.layer6(c6)
        c5_lat = self.latlayer_fc(c5)
        c6_lat = self.latlayer_c6(c6)
        c7_lat = self.latlayer_c7(c7)
        c4_fuse = self.conv5_ct_py(c5_lat, c4)
        c3_fuse = self.conv4_ct_py(c4_fuse, c3)
        c2_fuse = self.conv3_ct_py(c3_fuse, c2)
        c2_fuse = self.smooth_c3(c2_fuse)
        c3_fuse = self.smooth_c4(c3_fuse)
        c4_fuse = self.smooth_c5(c4_fuse)
        c2_fuse = self.conv2_SSH(c2_fuse)
        sources.append(c2_fuse)
        c3_fuse = self.conv3_SSH(c3_fuse)
        sources.append(c3_fuse)
        c4_fuse = self.conv4_SSH(c4_fuse)
        sources.append(c4_fuse)
        c5_lat = self.conv5_SSH(c5_lat)
        sources.append(c5_lat)
        c6_lat = self.conv6_SSH(c6_lat)
        sources.append(c6_lat)
        c7_lat = self.conv7_SSH(c7_lat)
        sources.append(c7_lat)
        prior_boxs = []
        prior_head_boxes = []
        prior_body_boxes = []
        for idx, f_layer in enumerate(sources):
            prior_boxs.append(self.priorbox.forward(idx, f_layer.shape[3],
                f_layer.shape[2]))
            if idx > 0:
                prior_head_boxes.append(self.priorbox_head.forward(idx - 1,
                    f_layer.shape[3], f_layer.shape[2]))
        self.priors = Variable(torch.cat([p for p in prior_boxs], 0),
            volatile=True)
        self.priors_head = Variable(torch.cat([p for p in prior_head_boxes],
            0), volatile=True)
        for idx, (x, l, c) in enumerate(zip(sources, self.face_loc, self.
            face_conf)):
            if idx == 0:
                tmp_conf = c(x)
                a, b, c, pos_conf = tmp_conf.chunk(4, 1)
                neg_conf = torch.cat([a, b, c], 1)
                max_conf, _ = neg_conf.max(1)
                max_conf = max_conf.view_as(pos_conf)
                conf.append(torch.cat([max_conf, pos_conf], 1).permute(0, 2,
                    3, 1).contiguous())
            else:
                tmp_conf = c(x)
                neg_conf, a, b, c = tmp_conf.chunk(4, 1)
                pos_conf = torch.cat([a, b, c], 1)
                max_conf, _ = pos_conf.max(1)
                max_conf = max_conf.view_as(neg_conf)
                conf.append(torch.cat([neg_conf, max_conf], 1).permute(0, 2,
                    3, 1).contiguous())
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
        for idx, (x, l, c) in enumerate(zip(sources[1:], self.head_loc,
            self.head_conf)):
            head_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            head_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        head_loc = torch.cat([o.view(o.size(0), -1) for o in head_loc], 1)
        head_conf = torch.cat([o.view(o.size(0), -1) for o in head_conf], 1)
        if self.phase == 'test':
            output = self.detect(loc.view(loc.size(0), -1, 4), self.softmax
                (conf.view(conf.size(0), -1, 2)), self.priors.type(type(x.
                data)))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), 
                -1, 2), self.priors, head_loc.view(head_loc.size(0), -1, 4
                ), head_conf.view(head_conf.size(0), -1, 2), self.priors_head
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            None
            pretrained_model = torch.load(base_file, map_location=lambda
                storage, loc: storage)
            model_dict = self.state_dict()
            pretrained_model = {k: v for k, v in pretrained_model.items() if
                k in model_dict}
            model_dict.update(pretrained_model)
            self.load_state_dict(model_dict)
            None
        else:
            None


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Goingqs_PyramidBox(_paritybench_base):
    pass
    def test_000(self):
        self._check(L2Norm(*[], **{'n_channels': 4, 'scale': 1.0}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(ConvBN(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(SSHContext(*[], **{'channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(Bottleneck(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

