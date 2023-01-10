import sys
_module = sys.modules[__name__]
del sys
demo = _module
conf = _module
setup = _module
ssds = _module
core = _module
checkpoint = _module
config = _module
criterion = _module
data_parallel = _module
evaluation_metrics = _module
optimizer = _module
tools = _module
visualize_funcs = _module
dataset = _module
coco = _module
dali_coco = _module
dali_dataiterator = _module
dali_tfrecord = _module
dataset_factory = _module
detection_dataset = _module
transforms = _module
modeling = _module
layers = _module
basic_layers = _module
box = _module
decoder = _module
layers_parser = _module
rfb_layers = _module
model_builder = _module
nets = _module
darknet = _module
densenet = _module
efficientnet = _module
effnet = _module
inception_v2 = _module
mobilenet = _module
regnet = _module
resnet = _module
rutils = _module
shufflenet = _module
bifpn = _module
fcos = _module
fpn = _module
fssd = _module
shelf = _module
ssd = _module
ssdsbase = _module
yolo = _module
pipeline = _module
pipeline_anchor_apex = _module
pipeline_anchor_basic = _module
ssds = _module
utils = _module
export = _module
train = _module
train_ddp = _module
visualize = _module

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


from collections import OrderedDict


import torch.nn as nn


import torch.nn.functional as F


import math


from torch.nn.parallel import DataParallel


from torch.nn.parallel._functions import Scatter


from torch.nn.parallel.parallel_apply import parallel_apply


import numpy as np


import torch.optim as optim


from torch.optim import lr_scheduler


from scipy.optimize import linear_sum_assignment


import torch.utils.data as data


import copy


from torchvision import transforms


import types


from numpy import random


import re


from torchvision.models import densenet


import torch.utils.model_zoo as model_zoo


import torchvision


from torchvision.models import mobilenet


from torchvision.models import resnet


from torchvision.models import shufflenetv2


import time


import torch.backends.cudnn as cudnn


from torch.utils.tensorboard import SummaryWriter


class MultiBoxLoss(nn.Module):
    """The MultiBox Loss is used to calculate the classification loss in object detection task.

    MultiBox Loss is introduce by [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325v5) and can be described as:

    .. math::
        L(x,c,l,g) = (Lconf(x, c) + \\alpha Lloc(x,l,g)) / N

    where, :math:`Lconf` is the CrossEntropy Loss and :math:`Lloc` is the SmoothL1 Loss
    weighted by :math:`\\alpha` which is set to 1 by cross val.

    Compute Targets:

    * Produce Confidence Target
        Indices by matching ground truth boxes
        with (default) 'priorboxes' that have jaccard index > threshold parameter
        (default threshold: 0.5).
    * Produce localization target 
        by 'encoding' variance into offsets of ground
        truth boxes and their matched  'priorboxes'.
    * Hard negative mining 
        to filter the excessive number of negative examples
        that comes with using a large number of default bounding boxes.
        (default negative:positive ratio 3:1)
    
    To reduce the code and make it more easier to embed into the pipeline. Here, only the classification loss is included in this class
    
    Args:
        negpos_ratio: ratio of negative over positive samples in the given feature map, Default: 3
    """

    def __init__(self, negpos_ratio=3, **kwargs):
        super(MultiBoxLoss, self).__init__()
        self.negpos_ratio = negpos_ratio

    def forward(self, pred_logits, target, depth):
        """
        Args:
            pred_logits: Predict class for each box
            target: Target class for each box
            depth: the sign for the positive and negative samples from anchor mathcing.                 Basically it can be splited to 3 types: positive(>0), background/negative(=0), ignore(<0)
        Returns:
            The classification loss for the given feature map
        """
        pred = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        max_ce = ce.max(2)[0].view(ce.shape[0], -1)
        depth_v = depth.view(ce.shape[0], -1)
        max_ce[depth_v != 0] = 0
        _, idx = max_ce.sort(1, descending=True)
        _, idx_rank = idx.sort(1)
        num_pos = (depth_v > 0).sum(1)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=depth_v.shape[1] - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        neg = neg.view_as(depth)
        return ce * ((depth > 0) + neg).gt(0).expand_as(ce)


class FocalLoss(nn.Module):
    """The Focal Loss is used to calculate the classification loss in object detection task.
    
    Focal Loss is introduce by [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) and can be described as:

    .. math::
        FL(p_t)=-\\alpha(1-p_t)^{\\gamma}ln(p_t)
    
    where :math:`p_t` is the cross entropy for each box. :math:`\\alpha` controls the ratio of positive sample and the :math:`\\gamma`
    controls the attention for the difficult samples.

    Args:
        alpha (float) : the param to control the ratio of positive sample, (0,1). Default: 0.25
        gamma (float) : the param to the attention for the difficult samples, [0,n), [0,5] has been shown in the original paper. Default: 2
    """

    def __init__(self, alpha=0.25, gamma=2, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, target, depth):
        """
        Args:
            pred_logits: Predict class for each box
            target: Target class for each box
            depth: Does not used in this function
        Returns:
            The classification loss for the given feature map
        """
        pred = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        alpha = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        pt = torch.where(target == 1, pred, 1 - pred)
        return alpha * (1.0 - pt) ** self.gamma * ce


class SmoothL1Loss(nn.Module):
    """The SmoothL1 Loss is used to calculate the localization loss in object detection task.
    
    This criterion that uses a squared term if the absolute
    element-wise error falls below 1 and an L1 term otherwise.
    It is less sensitive to outliers than the `MSELoss` and in some cases
    prevents exploding gradients (e.g. see `Fast R-CNN` paper by Ross Girshick).
    Also known as the Huber loss:

    .. math::
        \\text{loss}(x_i, y_i) =
        \\begin{cases}
        0.5 (x_i - y_i)^2, & \\text{if } |x_i - y_i| < \\beta \\\\
        |x_i - y_i| - 0.5, & \\text{otherwise }
        \\end{cases}
    
    :math:`x` and :math:`y` arbitrary shapes with a total of :math:`n` elements each
    the sum operation still operates over all the elements, and divides by :math:`n`.

    :math:`\\beta` is used as the threshold and smooth the loss

    Args:
        beta (float) : the param to control the threshold and smooth the loss, (0,1). Default: 0.11
    """

    def __init__(self, beta=0.11):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        """
        Args:
            pred: Predict box for each box
            target: Target box for each box
        Returns:
            The localization loss for the given feature map
        """
        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x ** 2 / self.beta
        return torch.where(x >= self.beta, l1, l2)


class IOULoss(nn.Module):
    """The IOU Loss is used to calculate the localization loss in object detection task.

    IoU Loss is introduce by [IoU Loss for 2D/3D Object Detection](https://arxiv.org/abs/1908.03851v1) and can be described as:

    .. math::
        IoU(A, B) = \\frac{A \\cap B}{A \\cup B} = \\frac{A \\cap B}{|A| + |B| - A \\cap  B}
    
    where, A and B represents the two convex shapes. In here, it means the predict box and the groundtruth box.

    This class actually implemented multiple IoU related losses and use :attr:`loss_type` to choose the specific loss func.

    Args:
        loss_type (str): param to choose the specific loss type.
    """

    def __init__(self, loss_type='iou'):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target):
        """
        Args:
            pred: Predict box for each box, format with x,y,w,h
            target: Target box for each box, format with x,y,w,h
        Returns:
            The localization loss for the given feature map
        """
        pred_lt, pred_rb, pred_wh = self.delta2ltrb(pred)
        target_lt, target_rb, target_wh = self.delta2ltrb(target)
        lt = torch.max(pred_lt, target_lt)
        rb = torch.min(pred_rb, target_rb)
        area_i = torch.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
        area_a = torch.prod(pred_wh, axis=2)
        area_b = torch.prod(target_wh, axis=2)
        area_union = area_a + area_b - area_i
        iou = (area_i + 1e-07) / (area_union + 1e-07)
        if self.loss_type == 'iou':
            iou = torch.clamp(iou, min=0, max=1.0).unsqueeze(2)
            return 1 - iou
        outer_lt = torch.min(pred_lt, target_lt)
        outer_rb = torch.max(pred_rb, target_rb)
        if self.loss_type == 'giou':
            area_outer = torch.prod(outer_rb - outer_lt, axis=2) * (outer_lt < outer_rb).all(axis=2) + 1e-07
            giou = iou - (area_outer - area_union) / area_outer
            giou = torch.clamp(giou, min=-1.0, max=1.0).unsqueeze(2)
            return 1 - giou
        inter_diag = ((pred[:, :, :2] - target[:, :, :2]) ** 2).sum(dim=2)
        outer_diag = ((outer_rb - outer_lt) ** 2).sum(dim=2) + 1e-07
        if self.loss_type == 'diou':
            diou = iou - inter_diag / outer_diag
            diou = torch.clamp(diou, min=-1.0, max=1.0).unsqueeze(2)
            return 1 - diou
        if self.loss_type == 'ciou':
            v = 4 / math.pi ** 2 * torch.pow(torch.atan(target_wh[:, :, 0] / target_wh[:, :, 1]) - torch.atan(pred_wh[:, :, 0] / pred_wh[:, :, 1]), 2)
            with torch.no_grad():
                S = 1 - iou
                alpha = v / (S + v)
            ciou = iou - (inter_diag / outer_diag + alpha * v)
            ciou = torch.clamp(ciou, min=-1.0, max=1.0).unsqueeze(2)
            return 1 - ciou

    def delta2ltrb(self, deltas):
        """ deltas [x,y,w,h] with [batch, anchor, 4, h, w]
        """
        pred_ctr = deltas[:, :, :2]
        pred_wh = torch.exp(deltas[:, :, 2:])
        return pred_ctr - 0.5 * pred_wh, pred_ctr + 0.5 * pred_wh, pred_wh


def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    """
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except:
                None
                None
                None
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class BalancedDataParallel(DataParallel):
    """ This class is used to replace the original pytorch DataParallel and balance the first GPU memory usage.

    The original script is from: https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/utils/data_parallel.py
    """

    def __init__(self, gpu0_bsz, *args, **kwargs):
        self.gpu0_bsz = gpu0_bsz
        super().__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        if self.gpu0_bsz == 0:
            replicas = replicas[1:]
        outputs = self.parallel_apply(replicas, device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids[:len(inputs)])

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].size(self.dim)
        num_dev = len(self.device_ids)
        gpu0_bsz = self.gpu0_bsz
        bsz_unit = (bsz - gpu0_bsz) // (num_dev - 1)
        if gpu0_bsz < bsz_unit:
            chunk_sizes = [gpu0_bsz] + [bsz_unit] * (num_dev - 1)
            delta = bsz - sum(chunk_sizes)
            for i in range(delta):
                chunk_sizes[i + 1] += 1
            if gpu0_bsz == 0:
                chunk_sizes = chunk_sizes[1:]
        else:
            return super().scatter(inputs, kwargs, device_ids)
        return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)


class SepConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, expand_ratio=1):
        padding = (kernel_size - 1) // 2
        super(SepConvBNReLU, self).__init__(nn.Conv2d(in_planes, in_planes, kernel_size, stride, padding, groups=in_planes, bias=False), nn.BatchNorm2d(in_planes), nn.ReLU6(inplace=True), nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False), nn.BatchNorm2d(out_planes), nn.ReLU6(inplace=True))


def ConvBNReLU(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class ConvBNReLUx2(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLUx2, self).__init__(nn.Conv2d(in_planes, out_planes // 2, 1, bias=False), nn.BatchNorm2d(out_planes // 2), nn.ReLU(inplace=True), nn.Conv2d(out_planes // 2, out_planes, kernel_size, stride, padding=padding, bias=False), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))


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


def Conv1x1BNReLU(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


def Conv3x3BNReLU(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class Residual(nn.Module):

    def __init__(self, nchannels):
        super(Residual, self).__init__()
        mid_channels = nchannels // 2
        self.conv1x1 = Conv1x1BNReLU(in_channels=nchannels, out_channels=mid_channels)
        self.conv3x3 = Conv3x3BNReLU(in_channels=mid_channels, out_channels=nchannels)

    def forward(self, x):
        out = self.conv3x3(self.conv1x1(x))
        return out + x


class DarkNet(nn.Module):

    def __init__(self, layers=[1, 2, 8, 8, 4], outputs=[5], groups=1, width_per_group=64, url=None):
        super(DarkNet, self).__init__()
        self.outputs = outputs
        self.url = url
        self.conv1 = Conv3x3BNReLU(in_channels=3, out_channels=32)
        self.block1 = self._make_layers(in_channels=32, out_channels=64, block_num=layers[0])
        self.block2 = self._make_layers(in_channels=64, out_channels=128, block_num=layers[1])
        self.block3 = self._make_layers(in_channels=128, out_channels=256, block_num=layers[2])
        self.block4 = self._make_layers(in_channels=256, out_channels=512, block_num=layers[3])
        self.block5 = self._make_layers(in_channels=512, out_channels=1024, block_num=layers[4])

    def _make_layers(self, in_channels, out_channels, block_num):
        _layers = []
        _layers.append(Conv3x3BNReLU(in_channels=in_channels, out_channels=out_channels, stride=2))
        for _ in range(block_num):
            _layers.append(Residual(nchannels=out_channels))
        return nn.Sequential(*_layers)

    def initialize(self):
        pass

    def forward(self, x):
        outputs = []
        x = self.conv1(x)
        for level in range(1, 6):
            if level > max(self.outputs):
                break
            x = getattr(self, 'block{}'.format(level))(x)
            if level in self.outputs:
                outputs.append(x)
        return outputs


class DenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, memory_efficient=False, outputs=[], url=None):
        super(DenseNet, self).__init__()
        self.url = url
        self.outputs = outputs
        self.block_config = block_config
        self.conv1 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)), ('norm', nn.BatchNorm2d(num_init_features)), ('relu', nn.ReLU(inplace=True)), ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = densenet._DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
            self.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = densenet._Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def initialize(self):
        if self.url:
            checkpoint = model_zoo.load_url(self.url)
            pattern = re.compile('^(.*denselayer\\d+\\.(?:norm|relu|conv))\\.((?:[12])\\.(?:weight|bias|running_mean|running_var))$')
            for key in list(checkpoint.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    checkpoint[new_key] = checkpoint[key]
                    del checkpoint[key]
            change_dict = {'features.conv0.': 'conv1.conv.', 'features.norm0.': 'conv1.norm.'}
            for i, num_layers in enumerate(self.block_config):
                change_dict['features.denseblock{}.'.format(i + 1)] = 'denseblock{}.'.format(i + 1)
                change_dict['features.transition{}.'.format(i + 1)] = 'transition{}.'.format(i + 1)
            for k, v in list(checkpoint.items()):
                for _k, _v in list(change_dict.items()):
                    if _k in k:
                        new_key = k.replace(_k, _v)
                        checkpoint[new_key] = checkpoint.pop(k)
            remove_dict = ['classifier.', 'features.norm5.']
            for k, v in list(checkpoint.items()):
                for _k in remove_dict:
                    if _k in k:
                        checkpoint.pop(k)
            self.load_state_dict(checkpoint)

    def forward(self, x):
        x = self.conv1(x)
        outputs = []
        for j in range(len(self.block_config)):
            level = j + 1
            if level > max(self.outputs):
                break
            if level > 1:
                x = getattr(self, 'transition{}'.format(level - 1))(x)
            x = getattr(self, 'denseblock{}'.format(level))(x)
            if level in self.outputs:
                outputs.append(x)
        return outputs


class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):

    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):

    def __init__(self, in_planes, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_planes, reduced_dim, 1), Swish(), nn.Conv2d(reduced_dim, in_planes, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, expand_ratio, kernel_size, stride, reduction_ratio=4, drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))
        layers = []
        if in_planes != hidden_dim:
            layers += [ConvBNReLU(in_planes, hidden_dim, 1)]
        layers += [ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim), SqueezeExcitation(hidden_dim, reduced_dim), nn.Conv2d(hidden_dim, out_planes, 1, bias=False), nn.BatchNorm2d(out_planes)]
        self.conv = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x):
        if self.use_residual:
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)


def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))


def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))


class EfficientNet(nn.Module):

    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, num_classes=1000):
        super(EfficientNet, self).__init__()
        settings = [[1, 16, 1, 1, 3], [6, 24, 2, 2, 3], [6, 40, 2, 2, 5], [6, 80, 3, 2, 3], [6, 112, 3, 1, 5], [6, 192, 4, 2, 5], [6, 320, 1, 1, 3]]
        self.settings = settings
        out_channels = _round_filters(32, width_mult)
        self.conv1 = ConvBNReLU(3, out_channels, 3, stride=2)
        in_channels = out_channels
        for j, (t, c, n, s, k) in enumerate(settings):
            out_channels = _round_filters(c, width_mult)
            repeats = _round_repeats(n, depth_mult)
            stage = []
            for i in range(repeats):
                stride = s if i == 0 else 1
                stage += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k)]
                in_channels = out_channels
            self.add_module('stage{}'.format(j + 1), nn.Sequential(*stage))
        last_channels = _round_filters(1280, width_mult)
        self.head_conv = ConvBNReLU(in_channels, last_channels, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(last_channels, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        for j in range(len(self.setting)):
            x = getattr(self, 'stage{}'.format(j + 1))(x)
        x = self.head_conv(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


class EfficientEx(EfficientNet):

    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2, outputs=[7], url=None):
        super(EfficientEx, self).__init__(width_mult=width_mult, depth_mult=depth_mult, dropout_rate=dropout_rate)
        self.url = url
        self.outputs = outputs
        self.depth_mult = depth_mult

    def initialize(self):
        if self.url:
            checkpoint = model_zoo.load_url(self.url)
            change_dict = {'features.0.': 'conv1.'}
            f_idx = 1
            for j, (t, c, n, s, k) in enumerate(self.settings):
                repeats = _round_repeats(n, self.depth_mult)
                for i in range(repeats):
                    change_dict['features.{}.'.format(f_idx)] = 'stage{}.{}.'.format(j + 1, i)
                    f_idx += 1
            change_dict['features.{}.'.format(f_idx)] = 'head_conv.'
            for k, v in list(checkpoint.items()):
                for _k, _v in list(change_dict.items()):
                    if _k in k:
                        new_key = k.replace(_k, _v)
                        checkpoint[new_key] = checkpoint.pop(k)
            for k, v in list(checkpoint.items()):
                if 'conv' in k and 'se' not in k:
                    k_list = k.split('.')
                    if k_list[-2].isdigit() and k_list[-3] != 'conv':
                        k_list[-2] = str(int(k_list[-2]) - 1)
                        new_key = '.'.join(k_list)
                        checkpoint[new_key] = checkpoint.pop(k)
            self.load_state_dict(checkpoint)

    def forward(self, x):
        x = self.conv1(x)
        outputs = []
        for j in range(len(self.settings)):
            level = j + 1
            if level > max(self.outputs):
                break
            x = getattr(self, 'stage{}'.format(level))(x)
            if level in self.outputs:
                outputs.append(x)
        return outputs


class EffNet(nn.Module):

    def __init__(self, model_name, outputs, exportable=False, **kwargs):
        super(EffNet, self).__init__()
        self.outputs = outputs
        if exportable:
            geffnet.config.set_exportable(True)
            model = geffnet.create_model(model_name, **kwargs)
        else:
            model = torch.hub.load('rwightman/gen-efficientnet-pytorch', model_name, **kwargs)
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1
        for j in range(7):
            self.add_module('block{}'.format(j + 1), getattr(model.blocks, '{}'.format(j)))

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        outputs = []
        for level in range(1, 8):
            if level > max(self.outputs):
                break
            x = getattr(self, 'block{}'.format(level))(x)
            if level in self.outputs:
                outputs.append(x)
        return outputs

    def initialize(self):
        pass


class InceptionV2ModuleA(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleA, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1), ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, padding=1))
        self.branch3 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1), ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, padding=1), ConvBNReLU(in_channels=out_channels3, out_channels=out_channels3, kernel_size=3, padding=1))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


def ConvBNReLUFactorization(in_channels, out_channels, kernel_sizes, paddings):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes, stride=1, padding=paddings), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True), nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_sizes, stride=1, padding=paddings), nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True))


class InceptionV2ModuleB(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleB, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1), ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce, kernel_sizes=[1, 3], paddings=[0, 1]), ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[3, 1], paddings=[1, 0]))
        self.branch3 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=[3, 1], paddings=[1, 0]), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=[1, 3], paddings=[0, 1]), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=[3, 1], paddings=[1, 0]), ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_sizes=[1, 3], paddings=[0, 1]))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV2ModuleC(nn.Module):

    def __init__(self, in_channels, out_channels1, out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleC, self).__init__()
        self.branch1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels1, kernel_size=1)
        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1)
        self.branch2_conv2a = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[1, 3], paddings=[0, 1])
        self.branch2_conv2b = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=[3, 1], paddings=[1, 0])
        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels3reduce, kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, stride=1, padding=1)
        self.branch3_conv3a = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=[3, 1], paddings=[1, 0])
        self.branch3_conv3b = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=[1, 3], paddings=[0, 1])
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1(x)
        x2 = self.branch2_conv1(x)
        out2 = torch.cat([self.branch2_conv2a(x2), self.branch2_conv2b(x2)], dim=1)
        x3 = self.branch3_conv2(self.branch3_conv1(x))
        out3 = torch.cat([self.branch3_conv3a(x3), self.branch3_conv3b(x3)], dim=1)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class InceptionV3ModuleD(nn.Module):

    def __init__(self, in_channels, out_channels1reduce, out_channels1, out_channels2reduce, out_channels2):
        super(InceptionV3ModuleD, self).__init__()
        self.branch1 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels1reduce, kernel_size=1), ConvBNReLU(in_channels=out_channels1reduce, out_channels=out_channels1, kernel_size=3, stride=2, padding=1))
        self.branch2 = nn.Sequential(ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1), ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, stride=1, padding=1), ConvBNReLU(in_channels=out_channels2, out_channels=out_channels2, kernel_size=3, stride=2, padding=1))
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out


class InceptionV2(nn.Module):

    def __init__(self, outputs=[], url=None):
        super(InceptionV2, self).__init__()
        self.outputs = outputs
        self.url = url
        self.block1 = nn.Sequential(ConvBNReLU(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block2 = nn.Sequential(ConvBNReLU(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.block3 = nn.Sequential(InceptionV2ModuleA(in_channels=192, out_channels1=64, out_channels2reduce=64, out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=32), InceptionV2ModuleA(in_channels=256, out_channels1=64, out_channels2reduce=64, out_channels2=96, out_channels3reduce=64, out_channels3=96, out_channels4=64), InceptionV3ModuleD(in_channels=320, out_channels1reduce=128, out_channels1=160, out_channels2reduce=64, out_channels2=96))
        self.block4 = nn.Sequential(InceptionV2ModuleB(in_channels=576, out_channels1=224, out_channels2reduce=64, out_channels2=96, out_channels3reduce=96, out_channels3=128, out_channels4=128), InceptionV2ModuleB(in_channels=576, out_channels1=192, out_channels2reduce=96, out_channels2=128, out_channels3reduce=96, out_channels3=128, out_channels4=128), InceptionV2ModuleB(in_channels=576, out_channels1=160, out_channels2reduce=128, out_channels2=160, out_channels3reduce=128, out_channels3=128, out_channels4=128), InceptionV2ModuleB(in_channels=576, out_channels1=96, out_channels2reduce=128, out_channels2=192, out_channels3reduce=160, out_channels3=160, out_channels4=128), InceptionV3ModuleD(in_channels=576, out_channels1reduce=128, out_channels1=192, out_channels2reduce=192, out_channels2=256))
        self.block5 = nn.Sequential(InceptionV2ModuleC(in_channels=1024, out_channels1=352, out_channels2reduce=192, out_channels2=160, out_channels3reduce=160, out_channels3=112, out_channels4=128), InceptionV2ModuleC(in_channels=1024, out_channels1=352, out_channels2reduce=192, out_channels2=160, out_channels3reduce=192, out_channels3=112, out_channels4=128))

    def initialize(self):
        pass

    def forward(self, x):
        outputs = []
        for level in range(1, 6):
            if level > max(self.outputs):
                break
            x = getattr(self, 'block{}'.format(level))(x)
            if level in self.outputs:
                outputs.append(x)
        return outputs


class MobileNet(nn.Module):

    def __init__(self, num_classes=1000, width_mult=1.0, version='v1', round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNet, self).__init__()
        input_channel = 32
        if version == 'v2':
            settings = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
            last_channel = 1280
            layer = mobilenet.InvertedResidual
        elif version == 'v1':
            settings = [[1, 64, 1, 1], [1, 128, 2, 2], [1, 256, 2, 2], [1, 512, 6, 2], [1, 1024, 2, 2]]
            last_channel = 1024
            layer = SepConvBNReLU
        self.settings = settings
        self.version = version
        input_channel = mobilenet._make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = mobilenet._make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.conv1 = mobilenet.ConvBNReLU(3, input_channel, stride=2)
        for j, (t, c, n, s) in enumerate(settings):
            output_channel = mobilenet._make_divisible(c * width_mult, round_nearest)
            layers = []
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(layer(input_channel, output_channel, stride=stride, expand_ratio=t))
                input_channel = output_channel
            self.add_module('layer{}'.format(j + 1), nn.Sequential(*layers))
        if self.version == 'v2':
            self.head_conv = mobilenet.ConvBNReLU(input_channel, self.last_channel, kernel_size=1)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, num_classes))
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
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        for j in range(len(self.settings)):
            x = getattr(self, 'layer{}'.format(j + 1))(x)
        if self.version == 'v2':
            x = self.head_conv(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


class MobileNetEx(MobileNet):

    def __init__(self, width_mult=1.0, version='v1', outputs=[7], url=None):
        super(MobileNetEx, self).__init__(width_mult=width_mult, version=version)
        self.url = url
        self.outputs = outputs

    def initialize(self):
        if self.url:
            checkpoint = model_zoo.load_url(self.url)
            if self.version == 'v2':
                change_dict = {'features.0.': 'conv1.'}
                f_idx = 1
                for j, (t, c, n, s) in enumerate(self.settings):
                    for i in range(n):
                        change_dict['features.{}.'.format(f_idx)] = 'layer{}.{}.'.format(j + 1, i)
                        f_idx += 1
                change_dict['features.{}.'.format(f_idx)] = 'head_conv.'
                for k, v in list(checkpoint.items()):
                    for _k, _v in list(change_dict.items()):
                        if _k in k:
                            new_key = k.replace(_k, _v)
                            checkpoint[new_key] = checkpoint.pop(k)
            else:
                change_dict = {'features.Conv2d_0.conv.': 'conv1.'}
                f_idx = 1
                for j, (t, c, n, s) in enumerate(self.settings):
                    for i in range(n):
                        for z in range(2):
                            change_dict['features.Conv2d_{}.depthwise.{}'.format(f_idx, z)] = 'layer{}.{}.{}'.format(j + 1, i, z)
                            change_dict['features.Conv2d_{}.pointwise.{}'.format(f_idx, z)] = 'layer{}.{}.{}'.format(j + 1, i, z + 3)
                        f_idx += 1
                for k, v in list(checkpoint.items()):
                    for _k, _v in list(change_dict.items()):
                        if _k in k:
                            new_key = k.replace(_k, _v)
                            checkpoint[new_key] = checkpoint.pop(k)
                remove_dict = ['classifier.']
                for k, v in list(checkpoint.items()):
                    for _k in remove_dict:
                        if _k in k:
                            checkpoint.pop(k)
                org_checkpoint = self.state_dict()
                org_checkpoint.update(checkpoint)
                checkpoint = org_checkpoint
            self.load_state_dict(checkpoint)

    def forward(self, x):
        x = self.conv1(x)
        outputs = []
        for j in range(len(self.settings)):
            level = j + 1
            if level > max(self.outputs):
                break
            x = getattr(self, 'layer{}'.format(level))(x)
            if level in self.outputs:
                outputs.append(x)
        return outputs


class ResStemIN(nn.Module):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(w_out)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(self, in_w, out_w):
        super(SimpleStemIN, self).__init__()
        self.conv = nn.Conv2d(in_w, out_w, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_w)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, ReLU, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(nn.Conv2d(w_in, w_se, 1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(w_se, w_in, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class BottleneckTransform(nn.Module):
    """Bottlenect transformation: 1x1, 3x3 [+SE], 1x1"""

    def __init__(self, w_in, w_out, stride, bm, gw, se_r):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(w_out * bm))
        g = w_b // gw
        self.a = nn.Conv2d(w_in, w_b, 1, stride=1, padding=0, bias=False)
        self.a_bn = nn.BatchNorm2d(w_b)
        self.a_relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(w_b, w_b, 3, stride=stride, padding=1, groups=g, bias=False)
        self.b_bn = nn.BatchNorm2d(w_b)
        self.b_relu = nn.ReLU(inplace=True)
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(w_b, w_se)
        self.c = nn.Conv2d(w_b, w_out, 1, stride=1, padding=0, bias=False)
        self.c_bn = nn.BatchNorm2d(w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform"""

    def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None):
        super(ResBottleneckBlock, self).__init__()
        self.proj_block = w_in != w_out or stride != 1
        if self.proj_block:
            self.proj = nn.Conv2d(w_in, w_out, 1, stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(w_out)
        self.f = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class AnyHead(nn.Module):
    """AnyNet head: AvgPool, 1x1."""

    def __init__(self, w_in, nc):
        super(AnyHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        super(AnyStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            name = 'b{}'.format(i + 1)
            self.add_module(name, block_fun(b_w_in, w_out, b_stride, bm, gw, se_r))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class AnyNet(nn.Module):
    """AnyNet model."""

    def __init__(self, **kwargs):
        super(AnyNet, self).__init__()
        if kwargs:
            self._construct(stem_w=kwargs['stem_w'], ds=kwargs['ds'], ws=kwargs['ws'], ss=kwargs['ss'], bms=kwargs['bms'], gws=kwargs['gws'], se_r=kwargs['se_r'], nc=kwargs['nc'])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                zero_init_gamma = hasattr(m, 'final_bn') and m.final_bn
                m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

    def _construct(self, stem_w, ds, ws, ss, bms, gws, se_r, nc):
        bms = bms if bms else [None for _d in ds]
        gws = gws if gws else [None for _d in ds]
        stage_params = list(zip(ds, ws, ss, bms, gws))
        self.stem = SimpleStemIN(3, stem_w)
        prev_w = stem_w
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            name = 's{}'.format(i + 1)
            self.add_module(name, AnyStage(prev_w, w, s, d, ResBottleneckBlock, bm, gw, se_r))
            prev_w = w
        self.head = AnyHead(w_in=prev_w, nc=nc)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [(w != wp or r != rp) for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


class RegNet(AnyNet):
    """RegNet model."""

    def __init__(self, w_a, w_0, w_m, d, group_w, bot_mul, se_r=None, num_classes=1000, outputs=[4], url=None, **kwargs):
        ws, num_stages, _, _ = generate_regnet(w_a, w_0, w_m, d)
        s_ws, s_ds = get_stages_from_blocks(ws, ws)
        s_gs = [group_w for _ in range(num_stages)]
        s_bs = [bot_mul for _ in range(num_stages)]
        s_ss = [(2) for _ in range(num_stages)]
        s_ws, s_gs = adjust_ws_gs_comp(s_ws, s_bs, s_gs)
        kwargs = {'stem_w': 32, 'ds': s_ds, 'ws': s_ws, 'ss': s_ss, 'bms': s_bs, 'gws': s_gs, 'se_r': se_r, 'nc': num_classes}
        self.outputs = outputs
        self.url = url
        super(RegNet, self).__init__(**kwargs)

    def initialize(self):
        if self.url:
            self.load_state_dict(model_zoo.load_url(self.url)['model_state'])

    def forward(self, x):
        x = self.stem(x)
        outputs = []
        for i, layer in enumerate([self.s1, self.s2, self.s3, self.s4]):
            level = i + 1
            if level > max(self.outputs):
                break
            x = layer(x)
            if level in self.outputs:
                outputs.append(x)
        return outputs


class BiFPNModule(nn.Module):

    def __init__(self, channels, levels, init=0.5, block=ConvBNReLU):
        super(BiFPNModule, self).__init__()
        self.levels = levels
        self.w1 = nn.Parameter(torch.Tensor(2, levels).fill_(init))
        self.w2 = nn.Parameter(torch.Tensor(3, levels - 2).fill_(init))
        for i in range(levels - 1, 0, -1):
            self.add_module('top-down-{}'.format(i - 1), block(channels, channels))
        for i in range(0, levels - 1, 1):
            self.add_module('bottom-up-{}'.format(i + 1), block(channels, channels))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, xx):
        assert len(xx) == self.levels
        levels = self.levels
        w1 = F.relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + 1e-06
        w2 = F.relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + 1e-06
        xs = [[]] + [x for x in xx[1:-1]] + [[]]
        for i in range(levels - 1, 0, -1):
            xx[i - 1] = w1[0, i - 1] * xx[i - 1] + w1[1, i - 1] * F.interpolate(xx[i], scale_factor=2, mode='nearest')
            xx[i - 1] = getattr(self, 'top-down-{}'.format(i - 1))(xx[i - 1])
        for i in range(0, levels - 2, 1):
            xx[i + 1] = w2[0, i] * xx[i + 1] + w2[1, i] * F.max_pool2d(xx[i], kernel_size=2) + w2[2, i] * xs[i + 1]
            xx[i + 1] = getattr(self, 'bottom-up-{}'.format(i + 1))(xx[i + 1])
        xx[levels - 1] = w1[0, levels - 1] * xx[levels - 1] + w1[1, levels - 1] * F.max_pool2d(xx[levels - 2], kernel_size=2)
        xx[levels - 1] = getattr(self, 'bottom-up-{}'.format(levels - 1))(xx[levels - 1])
        return xx


class SharedHead(nn.Sequential):

    def __init__(self, out_planes):
        layers = []
        for _ in range(4):
            layers += [ConvBNReLU(256, 256, 3)]
        layers += [nn.Conv2d(256, out_planes, 3, padding=1)]
        super(SharedHead, self).__init__(*layers)


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
        for x, l, c in zip(sources, self.loc, self.conf):
            loc.append(l(x).view(x.size(0), 4, -1))
            conf.append(c(x).view(x.size(0), self.num_classes, -1))
        loc = torch.cat(loc, 2).contiguous()
        conf = torch.cat(conf, 2).contiguous()
        return loc, conf


class BasicConvWithUpSample(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=True):
        super(BasicConvWithUpSample, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x, up_size=None):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if up_size is not None:
            x = F.upsample(x, size=up_size, mode='bilinear')
        return x


class SharedBlock(nn.Module):
    """ The conv params in this block is shared
    """

    def __init__(self, planes):
        super(SharedBlock, self).__init__()
        self.planes = planes
        self.conv1 = nn.Conv2d(self.planes, self.planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=0.25)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = out + x
        return self.relu2(out)


class ShelfPyramid(nn.Module):

    def __init__(self, settings, conv=nn.ConvTranspose2d, block=SharedBlock):
        super().__init__()
        extra_args = {'padding': 1, 'bias': True} if conv == nn.ConvTranspose2d else {}
        for i, depth in enumerate(settings):
            if i == 0:
                self.add_module('block{}'.format(i), block(depth))
            else:
                self.add_module('block{}'.format(i), block(depth))
                self.add_module('conv{}'.format(i), conv(settings[i - 1], depth, kernel_size=3, stride=2, **extra_args))

    def forward(self, xx):
        out = []
        x = xx[0]
        for i in range(len(xx)):
            if i != 0:
                x = getattr(self, 'conv{}'.format(i))(x) + xx[i]
            x = getattr(self, 'block{}'.format(i))(x)
            out.append(x)
        return out[::-1]


class Head(nn.Sequential):

    def __init__(self, in_channels, out_planes):
        super(Head, self).__init__(ConvBNReLU(in_channels, in_channels, 3), nn.Conv2d(in_channels, out_planes, 3, padding=1))


class SSDSBase(nn.Module):
    """Base class for all ssds model.
    """

    def __init__(self, backbone, num_classes):
        super(SSDSBase, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes

    def initialize_prior(self, layer):
        pi = 0.01
        b = -math.log((1 - pi) / pi)
        nn.init.constant_(layer.bias, b)
        nn.init.normal_(layer.weight, std=0.01)

    def initialize_head(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, val=0)

    def initialize_extra(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, val=0)


class YOLOV3(SSDSBase):
    """YOLOv3: An Incremental Improvement
    See: https://arxiv.org/abs/1804.02767v1 for more details.


    Args:
        backbone: backbone layers for input
        extras: contains transforms and extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        num_classes: num of classes 
    """

    def __init__(self, backbone, extras, head, num_classes):
        super(YOLOV3, self).__init__(backbone, num_classes)
        self.transforms = nn.ModuleList(extras[0])
        self.extras = nn.ModuleList(extras[1])
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.initialize()

    def initialize(self):
        """
        :meta private:
        """
        self.backbone.initialize()
        self.transforms.apply(self.initialize_extra)
        self.extras.apply(self.initialize_extra)
        self.loc.apply(self.initialize_head)
        self.conf.apply(self.initialize_head)
        for c in self.conf:
            c[-1].apply(self.initialize_prior)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images.

        Return:
            When self.training==True, loc and conf for each anchor box;

            When self.training==False. loc and conf.sigmoid() for each anchor box;

            For each player, conf with shape [batch, num_anchor*num_classes, height, width];

            For each player, loc  with shape [batch, num_anchor*4, height, width].
        """
        loc, conf = [list() for _ in range(2)]
        features = self.backbone(x)
        x = features[-1]
        xx = features[-1]
        features_len = len(features)
        for i in range(len(features))[::-1]:
            if i != features_len - 1:
                xx = F.interpolate(self.transforms[i](xx), scale_factor=2)
                xx = torch.cat((features[i], xx), dim=1)
            xx = self.extras[i](xx)
            features[i] = xx
        for i, (l, c) in enumerate(zip(self.loc, self.conf)):
            if i < features_len:
                xx = features[i]
            elif i == features_len:
                xx = self.extras[i](x)
            else:
                xx = self.extras[i](xx)
            loc.append(l(xx))
            conf.append(c(xx))
        if not self.training:
            conf = [c.sigmoid() for c in conf]
        return tuple(loc), tuple(conf)

    @staticmethod
    def add_extras(feature_layer, mbox, num_classes):
        """Define and declare the extras, loc and conf modules for the yolo v3 model.

        The feature_layer is defined in cfg.MODEL.FEATURE_LAYER. For yolo v3 model can be int, list of int and str:

        * int
            The int in the feature_layer represents the output feature in the backbone.
        * list of int
            The list of int in the feature_layer represents the output feature in the backbone, the first int is the \\
            backbone output and the second int is the upsampling branch to fuse feature.
        * str
            The str in the feature_layer represents the extra layers append at the end of the backbone.

        Args:
            feature_layer: the feature layers with detection head, defined by cfg.MODEL.FEATURE_LAYER
            mbox: the number of boxes for each feature map
            num_classes: the number of classes, defined by cfg.MODEL.NUM_CLASSES
        """
        nets_outputs, transform_layers, extra_layers, loc_layers, conf_layers = [list() for _ in range(5)]
        last_int_layer = [layer for layer in feature_layer[0] if isinstance(layer, int)][-1]
        for layer, depth, box in zip(feature_layer[0], feature_layer[1], mbox):
            if isinstance(layer, int):
                nets_outputs.append(layer)
                if layer == last_int_layer:
                    if isinstance(depth, list):
                        extra_layers += [ConvBNReLUx2(depth[0], depth[1], 3)]
                    else:
                        extra_layers += [ConvBNReLUx2(depth, depth // 2, 3)]
                else:
                    prev_depth = feature_layer[1][feature_layer[0].index(layer) + 1]
                    if isinstance(depth, list):
                        transform_layers += [ConvBNReLU(prev_depth[1], depth[0] // 2, 3)]
                        extra_layers += [ConvBNReLUx2(int(depth[0] * 1.5), depth[1], 3)]
                    else:
                        transform_layers += [ConvBNReLU(prev_depth // 2, depth // 2, 3)]
                        extra_layers += [ConvBNReLUx2(int(depth * 1.5), depth // 2, 3)]
            elif layer == 'Conv:S':
                extra_layers += [ConvBNReLU(in_channels, depth, 3, stride=2)]
            else:
                raise ValueError(layer + ' does not support by YOLO')
            in_channels = depth[1] if isinstance(depth, list) else depth // 2 if isinstance(layer, int) else depth
            loc_layers += [nn.Sequential(ConvBNReLU(in_channels, in_channels, 3), nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1))]
            conf_layers += [nn.Sequential(ConvBNReLU(in_channels, in_channels, 3), nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1))]
            in_channels = depth[0] if isinstance(depth, list) else depth
        return nets_outputs, (transform_layers, extra_layers), (loc_layers, conf_layers)


class SPPModule(nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPModule, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = [x]
        for i in range(self.num_levels):
            kernel_size = 4 * (i + 1) + 1
            padding = (kernel_size - 1) // 2
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=1)
        return x


class PANModule(nn.Module):

    def __init__(self, channels):
        super(PANModule, self).__init__()
        self.levels = len(channels)
        for i in range(self.levels - 1, 0, -1):
            self.add_module('top-down-{}-to-{}'.format(i, i - 1), ConvBNReLU(channels[i], channels[i - 1]))
            self.add_module('top-down-{}'.format(i - 1), ConvBNReLUx2(channels[i - 1] * 2, channels[i - 1]))
        for i in range(0, self.levels - 1, 1):
            self.add_module('bottom-up-{}-to-{}'.format(i, i + 1), ConvBNReLU(channels[i], channels[i + 1], stride=2))
            self.add_module('bottom-up-{}'.format(i + 1), ConvBNReLUx2(channels[i + 1] * 2, channels[i + 1]))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, xx):
        assert len(xx) == self.levels
        for i in range(self.levels - 1, 0, -1):
            xx[i - 1] = torch.cat((xx[i - 1], F.interpolate(getattr(self, 'top-down-{}-to-{}'.format(i, i - 1))(xx[i]), scale_factor=2, mode='nearest')), dim=1)
            xx[i - 1] = getattr(self, 'top-down-{}'.format(i - 1))(xx[i - 1])
        for i in range(0, self.levels - 1, 1):
            xx[i + 1] = torch.cat((xx[i + 1], getattr(self, 'bottom-up-{}-to-{}'.format(i, i + 1))(xx[i])), dim=1)
            xx[i + 1] = getattr(self, 'bottom-up-{}'.format(i + 1))(xx[i + 1])
        return xx


class YOLOV4(SSDSBase):
    """YOLO V4 Architecture
    See: https://arxiv.org/abs/2004.10934v1 for more details.

    Args:
        backbone: backbone layers for input
        extras: contains transforms, extra and fpn layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        num_classes: num of classes 
    """

    def __init__(self, backbone, extras, head, num_classes):
        super(YOLOV4, self).__init__(backbone, num_classes)
        self.transforms = nn.ModuleList(extras[0])
        self.extras = nn.ModuleList(extras[1])
        self.fpn = extras[2]
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.initialize()

    def initialize(self):
        """
        :meta private:
        """
        self.backbone.initialize()
        self.transforms.apply(self.initialize_extra)
        self.fpn.apply(self.initialize_extra)
        self.extras.apply(self.initialize_extra)
        self.loc.apply(self.initialize_head)
        self.conf.apply(self.initialize_head)
        for c in self.conf:
            c[-1].apply(self.initialize_prior)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images.

        Return:
            When self.training==True, loc and conf for each anchor box;

            When self.training==False. loc and conf.sigmoid() for each anchor box;

            For each player, conf with shape [batch, num_anchor*num_classes, height, width];

            For each player, loc  with shape [batch, num_anchor*4, height, width].
        """
        loc, conf = [list() for _ in range(2)]
        features = self.backbone(x)
        for i, t in enumerate(self.transforms):
            features[i] = t(features[i])
        features = self.fpn(features)
        x = features[-1]
        for e in self.extras:
            x = e(x)
            features.append(x)
        for i, (l, c) in enumerate(zip(self.loc, self.conf)):
            loc.append(l(features[i]))
            conf.append(c(features[i]))
        if not self.training:
            conf = [c.sigmoid() for c in conf]
        return tuple(loc), tuple(conf)

    @staticmethod
    def add_extras(feature_layer, mbox, num_classes):
        """Define and declare the extras, loc and conf modules for the yolo v4 model.

        The feature_layer is defined in cfg.MODEL.FEATURE_LAYER. For yolo v4 model can be int, list of int and str:

        * int
            The int in the feature_layer represents the output feature in the backbone.
        * list of int
            The list of int in the feature_layer represents the output feature in the backbone, the first int is the \\
            backbone output and the second int is the upsampling branch to fuse feature.
        * str
            The str in the feature_layer represents the extra layers append at the end of the backbone.

        Args:
            feature_layer: the feature layers with detection head, defined by cfg.MODEL.FEATURE_LAYER
            mbox: the number of boxes for each feature map
            num_classes: the number of classes, defined by cfg.MODEL.NUM_CLASSES
        """
        nets_outputs, transform_layers, extra_layers, loc_layers, conf_layers = [list() for _ in range(5)]
        last_int_layer = [layer for layer in feature_layer[0] if isinstance(layer, int)][-1]
        fpn_channels = []
        for layer, depth, box in zip(feature_layer[0], feature_layer[1], mbox):
            if isinstance(layer, int):
                nets_outputs.append(layer)
                fpn_channels.append(depth // 2)
                if layer == last_int_layer:
                    transform_layers += [nn.Sequential(ConvBNReLU(depth, depth // 2, 3), SPPModule(3), ConvBNReLU(depth * 2, depth // 2, 3))]
                else:
                    transform_layers += [ConvBNReLU(depth, depth // 2, 3)]
            elif layer == 'Conv:S':
                extra_layers += [ConvBNReLU(in_channels, depth, 3, stride=2)]
            else:
                raise ValueError(layer + ' does not support by YOLO')
            in_channels = depth // 2 if isinstance(layer, int) else depth
            loc_layers += [nn.Sequential(ConvBNReLU(in_channels, in_channels, 3), nn.Conv2d(in_channels, box * 4, kernel_size=3, padding=1))]
            conf_layers += [nn.Sequential(ConvBNReLU(in_channels, in_channels, 3), nn.Conv2d(in_channels, box * num_classes, kernel_size=3, padding=1))]
        num_stack = 1 if len(feature_layer) == 2 else feature_layer[2]
        fpn = nn.Sequential(*[PANModule(fpn_channels) for _ in range(num_stack)])
        return nets_outputs, (transform_layers, extra_layers, fpn), (loc_layers, conf_layers)


def box2delta(boxes, anchors):
    """Convert boxes to deltas from anchors"""
    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh
    boxes_wh = boxes[:, 2:] - boxes[:, :2] + 1
    boxes_ctr = boxes[:, :2] + 0.5 * boxes_wh
    return torch.cat([(boxes_ctr - anchors_ctr) / anchors_wh, torch.log(boxes_wh / anchors_wh)], 1)


def get_sample_region(boxes, stride, anchor_points, radius=1.5):
    """
    This code is from
    https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
    maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
    """
    stride = stride * radius
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    center_boxes = torch.cat((center - stride, center + stride), dim=-1)
    lt = anchor_points[:, :, None, :] - torch.max(center_boxes[:, :2], boxes[:, :2])[None, None, :]
    rb = torch.min(center_boxes[:, 2:], boxes[:, 2:])[None, None, :] - anchor_points[:, :, None, :]
    center_boxes = torch.cat((lt, rb), -1)
    inside_boxes_mask = center_boxes.min(-1)[0] > 0
    return inside_boxes_mask


def snap_to_anchors_by_iou(boxes, size, stride, anchors, num_classes, match, center_sampling_radius, is_centerness, device):
    """Snap target boxes (x, y, w, h) to anchors by the iou between target boxes and anchors"""
    num_anchors = anchors.size()[0] if anchors is not None else 1
    width, height = int(size[0] / stride), int(size[1] / stride)
    if boxes.nelement() == 0:
        if is_centerness:
            return torch.zeros([num_anchors, num_classes, height, width], device=device), torch.zeros([num_anchors, 4, height, width], device=device), torch.zeros([num_anchors, 1, height, width], device=device), torch.zeros([num_anchors, 1, height, width], device=device)
        else:
            return torch.zeros([num_anchors, num_classes, height, width], device=device), torch.zeros([num_anchors, 4, height, width], device=device), torch.zeros([num_anchors, 1, height, width], device=device)
    boxes, classes = boxes.split(4, dim=1)
    match_threshold, unmatch_threshold = match
    x, y = torch.meshgrid([torch.arange(0, size[i], stride, device=device, dtype=classes.dtype) for i in range(2)])
    xyxy = torch.stack((x, y, x, y), 2).unsqueeze(0)
    anchors = anchors.view(-1, 1, 1, 4)
    anchors = (xyxy + anchors).contiguous().view(-1, 4)
    boxes = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:] - 1], 1)
    xy1 = torch.max(anchors[:, None, :2], boxes[:, :2])
    xy2 = torch.min(anchors[:, None, 2:], boxes[:, 2:])
    inter = torch.prod((xy2 - xy1 + 1).clamp(0), 2)
    boxes_area = torch.prod(boxes[:, 2:] - boxes[:, :2] + 1, 1)
    anchors_area = torch.prod(anchors[:, 2:] - anchors[:, :2] + 1, 1)
    overlap = inter / (anchors_area[:, None] + boxes_area - inter)
    overlap, indices = overlap.max(1)
    box_target = box2delta(boxes[indices], anchors)
    box_target = box_target.view(num_anchors, 1, width, height, 4)
    box_target = box_target.transpose(1, 4).transpose(2, 3)
    box_target = box_target.squeeze().contiguous()
    depth = torch.ones_like(overlap) * -1
    depth[overlap < unmatch_threshold] = 0
    depth[overlap >= match_threshold] = classes[indices][overlap >= match_threshold].squeeze() + 1
    depth = depth.view(num_anchors, width, height)
    if center_sampling_radius > 0:
        anchor_points = torch.stack((x, y), dim=2) + stride // 2
        inside_boxes_mask = get_sample_region(boxes, stride, anchor_points, center_sampling_radius).float().max(-1)[0]
        depth = torch.min(depth, inside_boxes_mask[None, ...])
    depth = depth.transpose(1, 2).contiguous()
    cls_target = torch.zeros((anchors.size()[0], num_classes + 1), device=device, dtype=boxes.dtype)
    if classes.nelement() == 0:
        classes = torch.LongTensor([num_classes], device=device).expand_as(indices)
    else:
        classes = classes[indices].long()
    classes = classes.view(-1, 1)
    classes[overlap < unmatch_threshold] = num_classes
    cls_target.scatter_(1, classes, 1)
    cls_target = cls_target[:, :num_classes].view(-1, 1, width, height, num_classes)
    cls_target = cls_target.transpose(1, 4).transpose(2, 3)
    cls_target = cls_target.squeeze().contiguous()
    if is_centerness:
        lt = torch.abs(box_target[:, :2] - 0.5 * torch.exp(box_target[:, 2:]))
        rb = torch.abs(box_target[:, :2] - 0.5 * torch.exp(box_target[:, 2:]))
        centerness = torch.sqrt(torch.prod(torch.min(lt, rb) / torch.max(lt, rb), dim=1))
        return cls_target.view(num_anchors, num_classes, height, width), box_target.view(num_anchors, 4, height, width), centerness.view(num_anchors, 1, height, width), depth.view(num_anchors, 1, height, width)
    else:
        return cls_target.view(num_anchors, num_classes, height, width), box_target.view(num_anchors, 4, height, width), depth.view(num_anchors, 1, height, width)


INF = 100000


def snap_to_anchors_by_scale(boxes, size, stride, anchors, num_classes, match, center_sampling_radius, is_centerness, device):
    """Snap target boxes (x, y, w, h) to anchors by the scale of target boxes"""
    num_anchors = anchors.size()[0] if anchors is not None else 1
    width, height = int(size[0] / stride), int(size[1] / stride)
    if boxes.nelement() == 0:
        if is_centerness:
            return torch.zeros([num_anchors, num_classes, height, width], device=device), torch.zeros([num_anchors, 4, height, width], device=device), torch.zeros([num_anchors, 1, height, width], device=device), torch.zeros([num_anchors, 1, height, width], device=device)
        else:
            return torch.zeros([num_anchors, num_classes, height, width], device=device), torch.zeros([num_anchors, 4, height, width], device=device), torch.zeros([num_anchors, 1, height, width], device=device)
    boxes, classes = boxes.split(4, dim=1)
    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    anchors_size = torch.sqrt(torch.prod(anchors_wh, dim=1)).unsqueeze(1).unsqueeze(2)
    lower_threshold = (match[0] * anchors_size).clamp(-1)
    upper_threshold = match[1] * anchors_size
    x, y = torch.meshgrid([torch.arange(0, size[i], stride, device=device, dtype=classes.dtype) for i in range(2)])
    xyxy = torch.stack((x, y, x, y), 2).unsqueeze(0)
    anchors = anchors.view(-1, 1, 1, 4)
    anchors = (xyxy + anchors).contiguous().view(-1, 4)
    anchor_points = torch.stack((x, y), dim=2) + stride // 2
    boxes = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:] - 1], 1)
    boxes_area = torch.sqrt(torch.prod(boxes[:, 2:] - boxes[:, :2] + 1, 1))
    if center_sampling_radius > 0:
        is_cared_in_the_level = (boxes_area >= lower_threshold) & (boxes_area <= upper_threshold)
        anchor_points = torch.stack((x, y), dim=2) + stride // 2
        inside_boxes_mask = get_sample_region(boxes, stride, anchor_points).view(-1, boxes.shape[0])
    else:
        anchor_points = (torch.stack((x, y), dim=2) + stride // 2).view(-1, 2)
        lt = anchor_points[:, None, :] - boxes[:, :2]
        rb = boxes[:, 2:] - anchor_points[:, None, :]
        box_target = torch.cat([lt, rb], dim=-1)
        max_box_target = box_target.max(dim=-1)[0]
        is_cared_in_the_level = (max_box_target >= lower_threshold) & (max_box_target <= upper_threshold)
        inside_boxes_mask = box_target.min(dim=-1)[0] > 0
    mask = (is_cared_in_the_level & inside_boxes_mask).view(-1, boxes.shape[0])
    boxes_area = boxes_area.repeat(mask.shape[0], 1)
    boxes_area[mask == 0] = INF
    mask, _ = mask.max(dim=1)
    min_area, indices = boxes_area.min(dim=1)
    box_target = box2delta(boxes[indices], anchors)
    box_target = box_target.view(num_anchors, 1, width, height, 4)
    box_target = box_target.transpose(1, 4).transpose(2, 3)
    box_target = box_target.squeeze().contiguous()
    depth = torch.ones_like(mask, dtype=classes.dtype) * -1
    depth[mask == 0] = 0
    depth[mask != 0] = classes[indices][mask != 0].squeeze() + 1
    depth = depth.view(num_anchors, width, height).transpose(1, 2).contiguous()
    cls_target = torch.zeros((anchors.size()[0], num_classes + 1), device=device, dtype=boxes.dtype)
    if classes.nelement() == 0:
        classes = torch.LongTensor([num_classes], device=device).expand_as(indices)
    else:
        classes = classes[indices].long()
    classes = classes.view(-1, 1)
    classes[mask == 0] = num_classes
    cls_target.scatter_(1, classes, 1)
    cls_target = cls_target[:, :num_classes].view(-1, 1, width, height, num_classes)
    cls_target = cls_target.transpose(1, 4).transpose(2, 3)
    cls_target = cls_target.squeeze().contiguous()
    if is_centerness:
        lt = torch.abs(box_target[:, :2] - 0.5 * torch.exp(deltas[:, 2:]))
        rb = torch.abs(box_target[:, :2] - 0.5 * torch.exp(deltas[:, 2:]))
        centerness = torch.sqrt(torch.prod(torch.min(lt, rb) / torch.max(lt, rb), dim=1))
        return cls_target.view(num_anchors, num_classes, height, width), box_target.view(num_anchors, 4, height, width), centerness.view(num_anchors, 1, height, width), depth.view(num_anchors, 1, height, width)
    else:
        return cls_target.view(num_anchors, num_classes, height, width), box_target.view(num_anchors, 4, height, width), depth.view(num_anchors, 1, height, width)


def extract_targets(targets, anchors, classes, stride, size, match=[0.5, 0.4], center_sampling_radius=0, is_centerness=False):
    """snap the targets to anchors"""
    cls_target, box_target, depth = [], [], []
    for target in targets:
        target = target[target[:, -1] > -1]
        if isinstance(match[0], float):
            snapped = snap_to_anchors_by_iou(target, [(s * stride) for s in size[::-1]], stride, anchors[stride], classes, match, center_sampling_radius, is_centerness, targets.device)
        elif isinstance(match[0], list):
            idx = list(anchors).index(stride)
            snapped = snap_to_anchors_by_scale(target, [(s * stride) for s in size[::-1]], stride, anchors[stride], classes, match[idx], center_sampling_radius, is_centerness, targets.device)
        else:
            raise ValueError('unvalidate match param')
        for l, s in zip((cls_target, box_target, depth), snapped):
            l.append(s)
    return torch.stack(cls_target), torch.stack(box_target), torch.stack(depth)


class ModelWithLossBasic(torch.nn.Module):
    """ Class use to help the gpu memory becomes more balance in ddp model
    """

    def __init__(self, model, cls_criterion, loc_criterion, num_classes, match, center_sampling_radius):
        super(ModelWithLossBasic, self).__init__()
        self.model = model
        self.cls_criterion = cls_criterion
        self.loc_criterion = loc_criterion
        self.num_classes = num_classes
        self.match = match
        self.center_radius = center_sampling_radius

    def forward(self, images, targets, anchors):
        """ 
        :meta private:
        """
        loc, conf = self.model(images)
        cls_losses, loc_losses, fg_targets = [], [], []
        for j, (stride, anchor) in enumerate(anchors.items()):
            size = conf[j].shape[-2:]
            conf_target, loc_target, depth = extract_targets(targets, anchors, self.num_classes, stride, size, self.match, self.center_radius)
            fg_targets.append((depth > 0).sum().float().clamp(min=1))
            c = conf[j].view_as(conf_target).float()
            cls_mask = (depth >= 0).expand_as(conf_target).float()
            cls_loss = self.cls_criterion(c, conf_target, depth)
            cls_loss = cls_mask * cls_loss
            cls_losses.append(cls_loss.sum())
            l = loc[j].view_as(loc_target).float()
            loc_loss = self.loc_criterion(l, loc_target)
            loc_mask = (depth > 0).expand_as(loc_loss).float()
            loc_loss = loc_mask * loc_loss
            loc_losses.append(loc_loss.sum())
        fg_targets = torch.stack(fg_targets).sum()
        cls_loss = torch.stack(cls_losses).sum() / fg_targets
        loc_loss = torch.stack(loc_losses).sum() / fg_targets
        return cls_loss, loc_loss, cls_losses, loc_losses


class ExportModel(nn.Module):

    def __init__(self, model, nhwc):
        super(ExportModel, self).__init__()
        self.model = model
        self.nhwc = nhwc

    def forward(self, x):
        if self.nhwc:
            x = x.permute(0, 3, 1, 2).contiguous() / 255.0
        return self.model(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AnyHead,
     lambda: ([], {'w_in': 4, 'nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AnyNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AnyStage,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'stride': 1, 'd': 4, 'block_fun': _mock_layer, 'bm': 4, 'gw': 4, 'se_r': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConvWithUpSample,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicSepConv,
     lambda: ([], {'in_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BottleneckTransform,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'stride': 1, 'bm': 4, 'gw': 4, 'se_r': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBNReLUx2,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DarkNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ExportModel,
     lambda: ([], {'model': _mock_layer(), 'nhwc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Head,
     lambda: ([], {'in_channels': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IOULoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (InceptionV2ModuleA,
     lambda: ([], {'in_channels': 4, 'out_channels1': 4, 'out_channels2reduce': 4, 'out_channels2': 4, 'out_channels3reduce': 4, 'out_channels3': 4, 'out_channels4': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionV2ModuleB,
     lambda: ([], {'in_channels': 4, 'out_channels1': 4, 'out_channels2reduce': 4, 'out_channels2': 4, 'out_channels3reduce': 4, 'out_channels3': 4, 'out_channels4': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionV2ModuleC,
     lambda: ([], {'in_channels': 4, 'out_channels1': 4, 'out_channels2reduce': 4, 'out_channels2': 4, 'out_channels3reduce': 4, 'out_channels3': 4, 'out_channels4': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionV3ModuleD,
     lambda: ([], {'in_channels': 4, 'out_channels1reduce': 4, 'out_channels1': 4, 'out_channels2reduce': 4, 'out_channels2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MemoryEfficientSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBottleneckBlock,
     lambda: ([], {'w_in': 4, 'w_out': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResStemIN,
     lambda: ([], {'w_in': 4, 'w_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Residual,
     lambda: ([], {'nchannels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SE,
     lambda: ([], {'w_in': 4, 'w_se': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SPPModule,
     lambda: ([], {'num_levels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SepConvBNReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SharedBlock,
     lambda: ([], {'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SharedHead,
     lambda: ([], {'out_planes': 4}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     True),
    (SimpleStemIN,
     lambda: ([], {'in_w': 4, 'out_w': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SmoothL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SqueezeExcitation,
     lambda: ([], {'in_planes': 4, 'reduced_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Swish,
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

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

