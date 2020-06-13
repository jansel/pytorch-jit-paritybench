import sys
_module = sys.modules[__name__]
del sys
data = _module
coco = _module
config = _module
data_augment = _module
voc0712 = _module
voc_eval = _module
layers = _module
functions = _module
detection = _module
prior_box = _module
modules = _module
multibox_loss = _module
RFB_Net_E_vgg = _module
RFB_Net_mobile = _module
RFB_Net_vgg = _module
models = _module
test_RFB = _module
train_RFB = _module
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


import torch.backends.cudnn as cudnn


from torch.autograd import Function


from torch.autograd import Variable


from math import sqrt as sqrt


from itertools import product as product


import torch.nn.functional as F


import numpy as np


import torch.utils.data as data


import torch.optim as optim


import torch.nn.init as init


import math


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
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 
        2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:,
        :2].unsqueeze(0).expand(A, B, 2))
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
    area_a = ((box_a[:, (2)] - box_a[:, (0)]) * (box_a[:, (3)] - box_a[:, (1)])
        ).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, (2)] - box_b[:, (0)]) * (box_b[:, (3)] - box_b[:, (1)])
        ).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


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
        bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
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
            match(self.threshold, truths, defaults, self.variance, labels,
                loc_t, conf_t, idx)
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
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view
            (-1, 1))
        loss_c[pos.view(-1, 1)] = 0
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
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01,
            affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8
        ):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(BasicConv(in_planes, 2 * inter_planes,
            kernel_size=1, stride=stride), BasicConv(2 * inter_planes, 2 *
            inter_planes, kernel_size=3, stride=1, padding=1, relu=False))
        self.branch1 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, 2 *
            inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)
            ), BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3,
            stride=1, padding=3, dilation=3, relu=False))
        self.branch2 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes //
            2 * 3, kernel_size=3, stride=1, padding=1), BasicConv(
            inter_planes // 2 * 3, 2 * inter_planes, kernel_size=3, stride=
            stride, padding=1), BasicConv(2 * inter_planes, 2 *
            inter_planes, kernel_size=3, stride=1, padding=5, dilation=5,
            relu=False))
        self.branch3 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes //
            2 * 3, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv
            (inter_planes // 2 * 3, 2 * inter_planes, kernel_size=(7, 1),
            stride=stride, padding=(3, 0)), BasicConv(2 * inter_planes, 2 *
            inter_planes, kernel_size=3, stride=1, padding=7, dilation=7,
            relu=False))
        self.ConvLinear = BasicConv(8 * inter_planes, out_planes,
            kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1,
            stride=stride, relu=False)
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


class BasicRFB_c(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8
        ):
        super(BasicRFB_c, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(BasicConv(in_planes, 2 * inter_planes,
            kernel_size=1, stride=stride), BasicConv(2 * inter_planes, 2 *
            inter_planes, kernel_size=3, stride=1, padding=1, relu=False))
        self.branch1 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, 2 *
            inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)
            ), BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3,
            stride=1, padding=3, dilation=3, relu=False))
        self.branch2 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes //
            2 * 3, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv
            (inter_planes // 2 * 3, 2 * inter_planes, kernel_size=(7, 1),
            stride=stride, padding=(3, 0)), BasicConv(2 * inter_planes, 2 *
            inter_planes, kernel_size=3, stride=1, padding=7, dilation=7,
            relu=False))
        self.ConvLinear = BasicConv(6 * inter_planes, out_planes,
            kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1,
            stride=stride, relu=False)
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


class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes,
            kernel_size=3, stride=1, padding=1, relu=False))
        self.branch1 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes,
            kernel_size=(3, 1), stride=1, padding=(1, 0)), BasicConv(
            inter_planes, inter_planes, kernel_size=3, stride=1, padding=3,
            dilation=3, relu=False))
        self.branch2 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes,
            kernel_size=(1, 3), stride=stride, padding=(0, 1)), BasicConv(
            inter_planes, inter_planes, kernel_size=3, stride=1, padding=3,
            dilation=3, relu=False))
        self.branch3 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes,
            kernel_size=(3, 1), stride=1, padding=(1, 0)), BasicConv(
            inter_planes, inter_planes, kernel_size=3, stride=1, padding=5,
            dilation=5, relu=False))
        self.branch4 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes,
            kernel_size=(1, 3), stride=stride, padding=(0, 1)), BasicConv(
            inter_planes, inter_planes, kernel_size=3, stride=1, padding=5,
            dilation=5, relu=False))
        self.branch5 = nn.Sequential(BasicConv(in_planes, inter_planes // 2,
            kernel_size=1, stride=1), BasicConv(inter_planes // 2, 
            inter_planes // 4 * 3, kernel_size=(1, 3), stride=1, padding=(0,
            1)), BasicConv(inter_planes // 4 * 3, inter_planes, kernel_size
            =(3, 1), stride=stride, padding=(1, 0)), BasicConv(inter_planes,
            inter_planes, kernel_size=3, stride=1, padding=7, dilation=7,
            relu=False))
        self.branch6 = nn.Sequential(BasicConv(in_planes, inter_planes // 2,
            kernel_size=1, stride=1), BasicConv(inter_planes // 2, 
            inter_planes // 4 * 3, kernel_size=(3, 1), stride=1, padding=(1,
            0)), BasicConv(inter_planes // 4 * 3, inter_planes, kernel_size
            =(1, 3), stride=stride, padding=(0, 1)), BasicConv(inter_planes,
            inter_planes, kernel_size=3, stride=1, padding=7, dilation=7,
            relu=False))
        self.ConvLinear = BasicConv(7 * inter_planes, out_planes,
            kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1,
            stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        x6 = self.branch6(x)
        out = torch.cat((x0, x1, x2, x3, x4, x5, x6), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out


class RFBNet(nn.Module):

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(RFBNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size
        if size == 300:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            None
            return
        self.base = nn.ModuleList(base)
        self.reduce = BasicConv(512, 256, kernel_size=1, stride=1)
        self.up_reduce = BasicConv(1024, 256, kernel_size=1, stride=1)
        self.Norm = BasicRFB_a(512, 512, stride=1, scale=1.0)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

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
        for k in range(23):
            x = self.base[k](x)
        s1 = self.reduce(x)
        for k in range(23, len(self.base)):
            x = self.base[k](x)
        s2 = self.up_reduce(x)
        s2 = F.upsample(s2, scale_factor=2, mode='bilinear', align_corners=True
            )
        s = torch.cat((s1, s2), 1)
        ss = self.Norm(s)
        sources.append(ss)
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k % 2 == 0:
                sources.append(x)
        for x, l, c in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == 'test':
            output = loc.view(loc.size(0), -1, 4), self.softmax(conf.view(-
                1, self.num_classes))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), 
                -1, self.num_classes)
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            None
            self.load_state_dict(torch.load(base_file))
            None
        else:
            None


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01,
            affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicSepConv(nn.Module):

    def __init__(self, in_planes, kernel_size, stride=1, padding=0,
        dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicSepConv, self).__init__()
        self.out_channels = in_planes
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=
            in_planes, bias=bias)
        self.bn = nn.BatchNorm2d(in_planes, eps=1e-05, momentum=0.01,
            affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch1 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes //
            2 * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)), BasicConv
            (inter_planes // 2 * 3, inter_planes // 2 * 3, kernel_size=(3, 
            1), stride=stride, padding=(1, 0)), BasicSepConv(inter_planes //
            2 * 3, kernel_size=3, stride=1, padding=3, dilation=3, relu=False))
        self.branch2 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes //
            2 * 3, kernel_size=3, stride=1, padding=1), BasicConv(
            inter_planes // 2 * 3, inter_planes // 2 * 3, kernel_size=3,
            stride=stride, padding=1), BasicSepConv(inter_planes // 2 * 3,
            kernel_size=3, stride=1, padding=5, dilation=5, relu=False))
        self.ConvLinear = BasicConv(3 * inter_planes, out_planes,
            kernel_size=1, stride=1, relu=False)
        if in_planes == out_planes:
            self.identity = True
        else:
            self.identity = False
            self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1,
                stride=stride, relu=False)
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


class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4
        self.branch0 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicSepConv(inter_planes,
            kernel_size=3, stride=1, padding=1, dilation=1, relu=False))
        self.branch1 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes,
            kernel_size=(3, 1), stride=1, padding=(1, 0)), BasicSepConv(
            inter_planes, kernel_size=3, stride=1, padding=3, dilation=3,
            relu=False))
        self.branch2 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes,
            kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicSepConv(inter_planes, kernel_size=3, stride=1, padding=3,
            dilation=3, relu=False))
        self.branch3 = nn.Sequential(BasicConv(in_planes, inter_planes // 2,
            kernel_size=1, stride=1), BasicConv(inter_planes // 2, 
            inter_planes // 4 * 3, kernel_size=(1, 3), stride=1, padding=(0,
            1)), BasicConv(inter_planes // 4 * 3, inter_planes, kernel_size
            =(3, 1), stride=stride, padding=(1, 0)), BasicSepConv(
            inter_planes, kernel_size=3, stride=1, padding=5, dilation=5,
            relu=False))
        self.ConvLinear = BasicConv(4 * inter_planes, out_planes,
            kernel_size=1, stride=1, relu=False)
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


class RFBNet(nn.Module):

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(RFBNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size
        if size == 300:
            self.indicator = 1
        else:
            None
            return
        self.base = nn.ModuleList(base)
        self.Norm = BasicRFB_a(512, 512, stride=1, scale=1.0)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

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
        for k in range(12):
            x = self.base[k](x)
        s = self.Norm(x)
        sources.append(s)
        for k in range(12, len(self.base)):
            x = self.base[k](x)
        sources.append(x)
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k % 2 == 0:
                sources.append(x)
        for x, l, c in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == 'test':
            output = loc.view(loc.size(0), -1, 4), self.softmax(conf.view(-
                1, self.num_classes))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), 
                -1, self.num_classes)
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            None
            self.load_state_dict(torch.load(base_file))
            None
        else:
            None


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01,
            affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(BasicConv(in_planes, 2 * inter_planes,
            kernel_size=1, stride=stride), BasicConv(2 * inter_planes, 2 *
            inter_planes, kernel_size=3, stride=1, padding=visual, dilation
            =visual, relu=False))
        self.branch1 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, 2 *
            inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)
            ), BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3,
            stride=1, padding=visual + 1, dilation=visual + 1, relu=False))
        self.branch2 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes //
            2 * 3, kernel_size=3, stride=1, padding=1), BasicConv(
            inter_planes // 2 * 3, 2 * inter_planes, kernel_size=3, stride=
            stride, padding=1), BasicConv(2 * inter_planes, 2 *
            inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1,
            dilation=2 * visual + 1, relu=False))
        self.ConvLinear = BasicConv(6 * inter_planes, out_planes,
            kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1,
            stride=stride, relu=False)
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


class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4
        self.branch0 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes,
            kernel_size=3, stride=1, padding=1, relu=False))
        self.branch1 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes,
            kernel_size=(3, 1), stride=1, padding=(1, 0)), BasicConv(
            inter_planes, inter_planes, kernel_size=3, stride=1, padding=3,
            dilation=3, relu=False))
        self.branch2 = nn.Sequential(BasicConv(in_planes, inter_planes,
            kernel_size=1, stride=1), BasicConv(inter_planes, inter_planes,
            kernel_size=(1, 3), stride=stride, padding=(0, 1)), BasicConv(
            inter_planes, inter_planes, kernel_size=3, stride=1, padding=3,
            dilation=3, relu=False))
        self.branch3 = nn.Sequential(BasicConv(in_planes, inter_planes // 2,
            kernel_size=1, stride=1), BasicConv(inter_planes // 2, 
            inter_planes // 4 * 3, kernel_size=(1, 3), stride=1, padding=(0,
            1)), BasicConv(inter_planes // 4 * 3, inter_planes, kernel_size
            =(3, 1), stride=stride, padding=(1, 0)), BasicConv(inter_planes,
            inter_planes, kernel_size=3, stride=1, padding=5, dilation=5,
            relu=False))
        self.ConvLinear = BasicConv(4 * inter_planes, out_planes,
            kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1,
            stride=stride, relu=False)
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


class RFBNet(nn.Module):
    """RFB Net for object detection
    The network is based on the SSD architecture.
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1711.07767.pdf for more details on RFB Net.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(RFBNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size
        if size == 300:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            None
            return
        self.base = nn.ModuleList(base)
        self.Norm = BasicRFB_a(512, 512, stride=1, scale=1.0)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

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
        for k in range(23):
            x = self.base[k](x)
        s = self.Norm(x)
        sources.append(s)
        for k in range(23, len(self.base)):
            x = self.base[k](x)
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k % 2 == 0:
                sources.append(x)
        for x, l, c in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == 'test':
            output = loc.view(loc.size(0), -1, 4), self.softmax(conf.view(-
                1, self.num_classes))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), 
                -1, self.num_classes)
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
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ruinmessi_RFBNet(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicConv(*[], **{'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BasicRFB(*[], **{'in_planes': 64, 'out_planes': 4}), [torch.rand([4, 64, 64, 64])], {})

    def test_002(self):
        self._check(BasicRFB_c(*[], **{'in_planes': 64, 'out_planes': 4}), [torch.rand([4, 64, 64, 64])], {})

    def test_003(self):
        self._check(BasicRFB_a(*[], **{'in_planes': 64, 'out_planes': 4}), [torch.rand([4, 64, 64, 64])], {})

    def test_004(self):
        self._check(BasicSepConv(*[], **{'in_planes': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

