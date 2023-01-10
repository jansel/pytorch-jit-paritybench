import sys
_module = sys.modules[__name__]
del sys
cc_attention = _module
functions = _module
dataset = _module
datasets = _module
engine = _module
evaluate = _module
loss = _module
criterion = _module
loss = _module
lovasz_losses = _module
networks = _module
ccnet = _module
deeplabv3 = _module
pspnet = _module
test = _module
train = _module
utils = _module
criterion = _module
encoding = _module
logger = _module
loss = _module
pyt_utils = _module
utils = _module

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


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import Softmax


import numpy as np


import random


import collections


import torchvision


from torch.utils import data


import time


import torch.distributed as dist


import scipy


from scipy import ndimage


import torch.backends.cudnn as cudnn


from torch.autograd import Variable


import torchvision.models as models


from collections import OrderedDict


import scipy.ndimage as nd


from math import ceil


import math


from torch.nn import functional as F


import torch.utils.model_zoo as model_zoo


import functools


import torch.optim as optim


import scipy.misc


import logging


from torch.autograd import Function


import torch.cuda.comm as comm


from torch.nn.parallel.data_parallel import DataParallel


from torch.nn.parallel.parallel_apply import get_a_var


from torch.nn.parallel._functions import ReduceAddCoalesced


from torch.nn.parallel._functions import Broadcast


from collections import defaultdict


def INF(B, H, W):
    return -torch.diag(torch.tensor(float('inf')).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        return self.gamma * (out_H + out_W) + x


class CriterionDSN(nn.Module):
    """
    DSN : We need to consider two supervision for the model.
    """

    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            None

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)
        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)
        return loss1 + loss2 * 0.4


class OhemCrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255, thresh=0.7, min_kept=100000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def find_threshold(self, np_predict, np_target):
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0 / factor, 1.0 / factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0 / factor, 1.0 / factor), order=0)
        n, c, h, w = predict.shape
        min_kept = self.min_kept // (factor * factor)
        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))
        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept) - 1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape
        threshold = self.find_threshold(np_predict, np_target)
        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))
        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            None
        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = torch.from_numpy(input_label.reshape(target.size())).long()
        return new_target

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)
        return self.criterion(predict, target)


class CriterionOhemDSN(nn.Module):
    """
    DSN : We need to consider two supervision for the model.
    """

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduce=True):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2d(ignore_index, thresh, min_kept)
        self.criterion2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            None

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)
        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)
        return loss1 + loss2 * 0.4


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes is 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes) for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


class CriterionOhemDSN2(nn.Module):
    """
    DSN : We need to consider two supervision for the model.
    """

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True, reduction='mean'):
        super(CriterionOhemDSN2, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)
        loss2 = lovasz_softmax(F.softmax(scale_pred, dim=1), target, ignore=self.ignore_index)
        return loss1 + loss2


class StableBCELoss(torch.nn.modules.Module):

    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
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
        out = out + residual
        out = self.relu_inplace(out)
        return out


class PSPModule(nn.Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False), InPlaceABNSync(out_features), nn.Dropout2d(0.1))

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class RCCAModule(nn.Module):

    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False), InPlaceABNSync(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False), InPlaceABNSync(inter_channels))
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels + inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False), InPlaceABNSync(out_channels), nn.Dropout2d(0.1), nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)
        output = self.bottleneck(torch.cat([x, output], 1))
        return output


affine_par = True


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, criterion):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))
        self.head = nn.Sequential(PSPModule(2048, 512), nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), InPlaceABNSync(512), nn.Dropout2d(0.1), nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.criterion = criterion

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))
        return nn.Sequential(*layers)

    def forward(self, x, labels=None):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_dsn = self.dsn(x)
        x = self.layer4(x)
        x = self.head(x)
        outs = [x, x_dsn]
        if self.criterion is not None and labels is not None:
            return self.criterion(outs, labels)
        else:
            return outs


class ASPPModule(nn.Module):
    """
    Reference: 
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=512, dilations=(12, 24, 36)):
        super(ASPPModule, self).__init__()
        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False), InPlaceABNSync(inner_features))
        self.conv2 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False), InPlaceABNSync(inner_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False), InPlaceABNSync(inner_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False), InPlaceABNSync(inner_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False), InPlaceABNSync(inner_features))
        self.bottleneck = nn.Sequential(nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False), InPlaceABNSync(out_features), nn.Dropout2d(0.1))

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        bottle = self.bottleneck(out)
        return bottle


class CriterionCrossEntropy(nn.Module):

    def __init__(self, ignore_index=255):
        super(CriterionCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.criterion(scale_pred, target)
        return loss


class CriterionOhemCrossEntropy(nn.Module):

    def __init__(self, ignore_index=255, thres=0.6, min_kept=200000):
        super(CriterionOhemCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = OhemCrossEntropy2d(ignore_index, thres, min_kept)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.criterion(scale_pred, target)
        return loss


class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created
    by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead
    of calling the callback of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]
    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


class DataParallelModel(DataParallel):
    """Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the
    batch dimension.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards pass, gradients from each replica are summed into the original module.
    Note that the outputs are not gathered, please use compatible
    :class:`encoding.parallel.DataParallelCriterion`.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is
    the same size (so that each GPU processes the same number of samples).

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> y = net(x)
    """

    def gather(self, outputs, output_device):
        return outputs

    def replicate(self, module, device_ids):
        modules = super(DataParallelModel, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


class Reduce(Function):

    @staticmethod
    def forward(ctx, *inputs):
        ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
        inputs = sorted(inputs, key=lambda i: i.get_device())
        return comm.reduce_add(inputs)

    @staticmethod
    def backward(ctx, gradOutput):
        return Broadcast.apply(ctx.target_gpus, gradOutput)


torch_ver = torch.__version__[:3]


def _criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None, devices=None):
    assert len(modules) == len(inputs)
    assert len(targets) == len(inputs)
    if kwargs_tup:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    lock = threading.Lock()
    results = {}
    if torch_ver != '0.3':
        grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, target, kwargs, device=None):
        if torch_ver != '0.3':
            torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            if not isinstance(input, tuple):
                input = input,
            with torch.device(device):
                output = module(*(input + target), **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e
    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=(i, module, input, target, kwargs, device)) for i, (module, input, target, kwargs, device) in enumerate(zip(modules, inputs, targets, kwargs_tup, devices))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])
    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


class DataParallelCriterion(DataParallel):
    """
    Calculate loss in multiple-GPUs, which balance the memory usage for
    Semantic Segmentation.

    The targets are splitted across the specified devices by chunking in
    the batch dimension. Please use together with :class:`encoding.parallel.DataParallelModel`.

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. “Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> criterion = encoding.nn.DataParallelCriterion(criterion, device_ids=[0, 1, 2])
        >>> y = net(x)
        >>> loss = criterion(y, target)
    """

    def forward(self, inputs, *targets, **kwargs):
        if not self.device_ids:
            return self.module(inputs, *targets, **kwargs)
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(inputs, *targets[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = _criterion_parallel_apply(replicas, inputs, targets, kwargs)
        return Reduce.apply(*outputs) / len(outputs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (StableBCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_speedinghzl_CCNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

