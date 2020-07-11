import sys
_module = sys.modules[__name__]
del sys
src = _module
convert_mobilenetv2 = _module
convert_xception65 = _module
dataset = _module
apolloscape = _module
cityscapes = _module
pascal_voc = _module
eval_cityscapes = _module
logger = _module
log = _module
plot = _module
binary = _module
dice_loss = _module
focal_loss = _module
lovasz_loss = _module
multi = _module
focal_loss = _module
lovasz_loss = _module
ohem_loss = _module
softiou_loss = _module
sym_loss = _module
models = _module
common = _module
decoder = _module
encoder = _module
ibn = _module
mobilenet = _module
net = _module
oc = _module
scse = _module
spp = _module
tta = _module
xception = _module
train = _module
utils = _module
custum_aug = _module
functional = _module
metrics = _module
optimizer = _module
preprocess = _module
scheduler = _module
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


import tensorflow as tf


import torch


import numpy as np


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torchvision import transforms


from functools import partial


import matplotlib


import matplotlib.pyplot as plt


import torch.nn as nn


import torch.nn.functional as F


from collections import OrderedDict


from torchvision import models


from torch import nn


from torch.nn import functional as F


import torch.optim as optim


class DiceLoss(nn.Module):

    def __init__(self, smooth=0, eps=1e-07):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, preds, labels):
        return 1 - (2 * torch.sum(preds * labels) + self.smooth) / (torch.sum(preds) + torch.sum(labels) + self.smooth + self.eps)


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -(1 - pt) ** self.gamma * self.alpha * logpt
        return loss


def hinge(pred, label):
    signs = 2 * label - 1
    errors = 1 - pred * signs
    return errors


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels, ignore_index):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\\infty and +\\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore_index: label to ignore
    """
    logits = logits.contiguous().view(-1)
    labels = labels.contiguous().view(-1)
    if ignore_index is not None:
        mask = labels != ignore_index
        logits = logits[mask]
        labels = labels[mask]
    errors = hinge(logits, labels)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.elu(errors_sorted) + 1, grad)
    return loss


class LovaszLoss(nn.Module):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\\infty and +\\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore_index: label to ignore
    """

    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        return lovasz_hinge_flat(logits, labels, self.ignore_index)


class MixedDiceBCELoss(nn.Module):

    def __init__(self, dice_weight=0.2, bce_weight=0.9):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, preds, labels):
        preds = torch.sigmoid(preds)
        loss = self.dice_weight * self.dice_loss(preds, labels) + self.bce_weight * self.bce_loss(preds, labels)
        return loss


class BinaryClassCriterion(nn.Module):

    def __init__(self, loss_type='BCE', **kwargs):
        super().__init__()
        if loss_type == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss(**kwargs)
        elif loss_type == 'Focal':
            self.criterion = FocalLoss(**kwargs)
        elif loss_type == 'Lovasz':
            self.criterion = LovaszLoss(**kwargs)
        elif loss_type == 'Dice':
            self.criterion = DiceLoss(**kwargs)
        elif loss_type == 'MixedDiceBCE':
            self.criterion = MixedDiceBCELoss(**kwargs)
        else:
            raise NotImplementedError

    def forward(self, preds, labels):
        loss = self.criterion(preds, labels)
        return loss


def lovasz_softmax_flat(prb, lbl, ignore_index, only_present):
    """
    Multi-class Lovasz-Softmax loss
      prb: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      lbl: [P] Tensor, ground truth labels (between 0 and C - 1)
      ignore_index: void class labels
      only_present: average only on classes present in ground truth
    """
    C = prb.shape[0]
    prb = prb.permute(1, 2, 0).contiguous().view(-1, C)
    lbl = lbl.view(-1)
    if ignore_index is not None:
        mask = lbl != ignore_index
        if mask.sum() == 0:
            return torch.mean(prb * 0)
        prb = prb[mask]
        lbl = lbl[mask]
    total_loss = 0
    cnt = 0
    for c in range(C):
        fg = (lbl == c).float()
        if only_present and fg.sum() == 0:
            continue
        errors = (fg - prb[:, (c)]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        total_loss += torch.dot(errors_sorted, lovasz_grad(fg_sorted))
        cnt += 1
    return total_loss / cnt


class LovaszSoftmax(nn.Module):
    """
    Multi-class Lovasz-Softmax loss
      logits: [B, C, H, W] class logits at each prediction (between -\\infty and \\infty)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      ignore_index: void class labels
      only_present: average only on classes present in ground truth
    """

    def __init__(self, ignore_index=None, only_present=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present

    def forward(self, logits, labels):
        probas = F.softmax(logits, dim=1)
        total_loss = 0
        batch_size = logits.shape[0]
        for prb, lbl in zip(probas, labels):
            total_loss += lovasz_softmax_flat(prb, lbl, self.ignore_index, self.only_present)
        return total_loss / batch_size


class OhemCrossEntropy2d(nn.Module):

    def __init__(self, thresh=0.6, min_kept=0, weight=None, ignore_index=255):
        super().__init__()
        self.ignore_label = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, predict, target):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
        """
        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))
        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            None
        elif num_valid > 0:
            prob = input_prob[:, (valid_flag)]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            None
        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        None
        target = torch.from_numpy(input_label.reshape(target.size())).long()
        return self.criterion(predict, target)


class SoftIoULoss(nn.Module):

    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, logit, target):
        N = len(logit)
        pred = F.softmax(logit, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)
        inter = pred * target_onehot
        inter = inter.view(N, self.n_classes, -1).sum(2)
        union = pred + target_onehot - pred * target_onehot
        union = union.view(N, self.n_classes, -1).sum(2)
        loss = inter / (union + 1e-16)
        return -loss.mean()


class MultiClassCriterion(nn.Module):

    def __init__(self, loss_type='CrossEntropy', **kwargs):
        super().__init__()
        if loss_type == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss(**kwargs)
        elif loss_type == 'Focal':
            self.criterion = FocalLoss(**kwargs)
        elif loss_type == 'Lovasz':
            self.criterion = LovaszSoftmax(**kwargs)
        elif loss_type == 'OhemCrossEntropy':
            self.criterion = OhemCrossEntropy2d(**kwargs)
        elif loss_type == 'SoftIOU':
            self.criterion = SoftIoULoss(**kwargs)
        else:
            raise NotImplementedError

    def forward(self, preds, labels):
        loss = self.criterion(preds, labels)
        return loss


class SoftCrossEntropy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, valid_mask=None):
        if valid_mask is not None:
            loss = 0
            batch_size = logits.shape[0]
            for logit, lbl, val_msk in zip(logits, labels, valid_mask):
                logit = logit[:, (val_msk)]
                lbl = lbl[:, (val_msk)]
                loss -= torch.mean(torch.mul(F.log_softmax(logit, dim=0), F.softmax(lbl, dim=0)))
            return loss / batch_size
        else:
            return torch.mean(torch.mul(F.log_softmax(logits, dim=1), F.softmax(labels, dim=1)))


class KlLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, valid_mask=None):
        if valid_mask is not None:
            loss = 0
            batch_size = logits.shape[0]
            for logit, lbl, val_msk in zip(logits, labels, valid_mask):
                logit = logit[:, (val_msk)]
                lbl = lbl[:, (val_msk)]
                loss += torch.mean(F.kl_div(F.log_softmax(logit, dim=0), F.softmax(lbl, dim=0), reduction='none'))
            return loss / batch_size
        else:
            return torch.mean(F.kl_div(F.log_softmax(logits, dim=1), F.softmax(labels, dim=1), reduction='none'))


class _ActivatedBatchNorm(nn.Module):

    def __init__(self, num_features, activation='relu', slope=0.01, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, **kwargs)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)
        elif activation == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x


class SeparableConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True):
        super().__init__()
        depthwise = nn.Conv2d(inplanes, inplanes, kernel_size, stride=stride, padding=dilation, dilation=dilation, groups=inplanes, bias=False)
        bn_depth = nn.BatchNorm2d(inplanes)
        pointwise = nn.Conv2d(inplanes, planes, 1, bias=False)
        bn_point = nn.BatchNorm2d(planes)
        if relu_first:
            self.block = nn.Sequential(OrderedDict([('relu', nn.ReLU()), ('depthwise', depthwise), ('bn_depth', bn_depth), ('pointwise', pointwise), ('bn_point', bn_point)]))
        else:
            self.block = nn.Sequential(OrderedDict([('depthwise', depthwise), ('bn_depth', bn_depth), ('relu1', nn.ReLU()), ('pointwise', pointwise), ('bn_point', bn_point), ('relu2', nn.ReLU())]))

    def forward(self, x):
        return self.block(x)


ActivatedBatchNorm = _ActivatedBatchNorm


class SCSEBlock(nn.Module):

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)), nn.ReLU(inplace=True), nn.Linear(int(channel // reduction), channel))
        self.spatial_se = nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        bahs, chs, _, _ = x.size()
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = torch.sigmoid(self.channel_excitation(chn_se).view(bahs, chs, 1, 1))
        chn_se = torch.mul(x, chn_se)
        spa_se = torch.sigmoid(self.spatial_se(x))
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)


class DecoderUnetSCSE(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1), ActivatedBatchNorm(middle_channels), SCSEBlock(middle_channels), nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1))

    def forward(self, *args):
        x = torch.cat(args, 1)
        return self.block(x)


class IBN(nn.Module):

    def __init__(self, planes):
        super().__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.Sequential(nn.InstanceNorm2d(half1, affine=True), nn.ReLU(inplace=True))
        self.BN = ActivatedBatchNorm(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class ImprovedIBNaDecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, 1), IBN(in_channels // 4), nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, stride=2, padding=1), ActivatedBatchNorm(in_channels // 4), nn.Conv2d(in_channels // 4, out_channels, 1), ActivatedBatchNorm(out_channels))

    def forward(self, x):
        return self.block(x)


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, int(channel / reduction), bias=False), nn.ReLU(inplace=True), nn.Linear(int(channel / reduction), channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DecoderUnetSEIBN(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(SELayer(in_channels), ImprovedIBNaDecoderBlock(in_channels, out_channels))

    def forward(self, *args):
        x = torch.cat(args, 1)
        return self.block(x)


class SelfAttentionBlock2D(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels is None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1), ActivatedBatchNorm(self.key_channels))
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels, kernel_size=1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = self.key_channels ** -0.5 * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=True)
        return context


class BaseOC_Context(nn.Module):
    """
    Output only the context features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout=0.05, sizes=(1,)):
        super().__init__()
        self.stages = nn.ModuleList([SelfAttentionBlock2D(in_channels, key_channels, value_channels, out_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0), ActivatedBatchNorm(out_channels), nn.Dropout2d(dropout))

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(context)
        return output


class BaseOC(nn.Module):

    def __init__(self, in_channels=2048, out_channels=256, dropout=0.05):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), ActivatedBatchNorm(out_channels), BaseOC_Context(in_channels=out_channels, out_channels=out_channels, key_channels=out_channels // 2, value_channels=out_channels // 2, dropout=dropout))

    def forward(self, x):
        return self.block(x)


class DecoderUnetOC(nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1), ActivatedBatchNorm(middle_channels), BaseOC(in_channels=middle_channels, out_channels=middle_channels, dropout=0.2), nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1))

    def forward(self, *args):
        x = torch.cat(args, 1)
        return self.block(x)


class DecoderSPP(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 48, 1, bias=False)
        self.bn = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.sep1 = SeparableConv2d(304, 256, relu_first=False)
        self.sep2 = SeparableConv2d(256, 256, relu_first=False)

    def forward(self, x, low_level_feat):
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        low_level_feat = self.conv(low_level_feat)
        low_level_feat = self.bn(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.sep1(x)
        x = self.sep2(x)
        return x


class ExpandedConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dilation=1, expand_ratio=6, skip_connection=False):
        super().__init__()
        self.stride = stride
        self.kernel_size = 3
        self.dilation = dilation
        self.expand_ratio = expand_ratio
        self.skip_connection = skip_connection
        middle_channels = in_channels * expand_ratio
        if self.expand_ratio != 1:
            self.expand = nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels, middle_channels, 1, bias=False)), ('bn', nn.BatchNorm2d(middle_channels)), ('relu', nn.ReLU6(inplace=True))]))
        self.depthwise = nn.Sequential(OrderedDict([('conv', nn.Conv2d(middle_channels, middle_channels, 3, stride, dilation, dilation, groups=middle_channels, bias=False)), ('bn', nn.BatchNorm2d(middle_channels)), ('relu', nn.ReLU6(inplace=True))]))
        self.project = nn.Sequential(OrderedDict([('conv', nn.Conv2d(middle_channels, out_channels, 1, bias=False)), ('bn', nn.BatchNorm2d(out_channels))]))

    def forward(self, x):
        if self.expand_ratio != 1:
            residual = self.project(self.depthwise(self.expand(x)))
        else:
            residual = self.project(self.depthwise(x))
        if self.skip_connection:
            outputs = x + residual
        else:
            outputs = residual
        return outputs


class MobileNetV2(nn.Module):

    def __init__(self, pretrained=False, model_path='../model/mobilenetv2_encoder/model.pth'):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU6()
        self.block0 = ExpandedConv(32, 16, expand_ratio=1)
        self.block1 = ExpandedConv(16, 24, stride=2)
        self.block2 = ExpandedConv(24, 24, skip_connection=True)
        self.block3 = ExpandedConv(24, 32, stride=2)
        self.block4 = ExpandedConv(32, 32, skip_connection=True)
        self.block5 = ExpandedConv(32, 32, skip_connection=True)
        self.block6 = ExpandedConv(32, 64)
        self.block7 = ExpandedConv(64, 64, dilation=2, skip_connection=True)
        self.block8 = ExpandedConv(64, 64, dilation=2, skip_connection=True)
        self.block9 = ExpandedConv(64, 64, dilation=2, skip_connection=True)
        self.block10 = ExpandedConv(64, 96, dilation=2)
        self.block11 = ExpandedConv(96, 96, dilation=2, skip_connection=True)
        self.block12 = ExpandedConv(96, 96, dilation=2, skip_connection=True)
        self.block13 = ExpandedConv(96, 160, dilation=2)
        self.block14 = ExpandedConv(160, 160, dilation=4, skip_connection=True)
        self.block15 = ExpandedConv(160, 160, dilation=4, skip_connection=True)
        self.block16 = ExpandedConv(160, 320, dilation=4)
        if pretrained:
            self.load_pretrained_model(model_path)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        return x

    def load_pretrained_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        None


class SegmentatorTTA(object):

    @staticmethod
    def hflip(x):
        return x.flip(3)

    @staticmethod
    def vflip(x):
        return x.flip(2)

    @staticmethod
    def trans(x):
        return x.transpose(2, 3)

    def pred_resize(self, x, size, net_type='unet'):
        h, w = size
        if net_type == 'unet':
            pred = self.forward(x)
            if x.shape[2:] == size:
                return pred
            else:
                return F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
        else:
            pred = self.forward(F.pad(x, (0, 1, 0, 1)))
            return F.interpolate(pred, size=(h + 1, w + 1), mode='bilinear', align_corners=True)[(...), :h, :w]

    def tta(self, x, scales=None, net_type='unet'):
        size = x.shape[2:]
        if scales is None:
            seg_sum = self.pred_resize(x, size, net_type)
            seg_sum += self.hflip(self.pred_resize(self.hflip(x), size, net_type))
            return seg_sum / 2
        else:
            seg_sum = self.pred_resize(x, size, net_type)
            seg_sum += self.hflip(self.pred_resize(self.hflip(x), size, net_type))
            for scale in scales:
                scaled = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=True)
                seg_sum += self.pred_resize(scaled, size, net_type)
                seg_sum += self.hflip(self.pred_resize(self.hflip(scaled), size, net_type))
            return seg_sum / ((len(scales) + 1) * 2)


def create_decoder(dec_type):
    if dec_type == 'unet_scse':
        return DecoderUnetSCSE
    elif dec_type == 'unet_seibn':
        return DecoderUnetSEIBN
    elif dec_type == 'unet_oc':
        return DecoderUnetOC
    else:
        raise NotImplementedError


class XceptionBlock(nn.Module):

    def __init__(self, channel_list, stride=1, dilation=1, skip_connection_type='conv', relu_first=True, low_feat=False):
        super().__init__()
        assert len(channel_list) == 4
        self.skip_connection_type = skip_connection_type
        self.relu_first = relu_first
        self.low_feat = low_feat
        if self.skip_connection_type == 'conv':
            self.conv = nn.Conv2d(channel_list[0], channel_list[-1], 1, stride=stride, bias=False)
            self.bn = nn.BatchNorm2d(channel_list[-1])
        self.sep_conv1 = SeparableConv2d(channel_list[0], channel_list[1], dilation=dilation, relu_first=relu_first)
        self.sep_conv2 = SeparableConv2d(channel_list[1], channel_list[2], dilation=dilation, relu_first=relu_first)
        self.sep_conv3 = SeparableConv2d(channel_list[2], channel_list[3], dilation=dilation, relu_first=relu_first, stride=stride)

    def forward(self, inputs):
        sc1 = self.sep_conv1(inputs)
        sc2 = self.sep_conv2(sc1)
        residual = self.sep_conv3(sc2)
        if self.skip_connection_type == 'conv':
            shortcut = self.conv(inputs)
            shortcut = self.bn(shortcut)
            outputs = residual + shortcut
        elif self.skip_connection_type == 'sum':
            outputs = residual + inputs
        elif self.skip_connection_type == 'none':
            outputs = residual
        else:
            raise ValueError('Unsupported skip connection type.')
        if self.low_feat:
            return outputs, sc2
        else:
            return outputs


class Xception65(nn.Module):

    def __init__(self, output_stride=8):
        super().__init__()
        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = 1, 2
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = 2, 4
        else:
            raise NotImplementedError
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = XceptionBlock([64, 128, 128, 128], stride=2)
        self.block2 = XceptionBlock([128, 256, 256, 256], stride=2, low_feat=True)
        self.block3 = XceptionBlock([256, 728, 728, 728], stride=entry_block3_stride)
        self.block4 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block5 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block6 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block7 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block8 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block9 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block10 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block11 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block12 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block13 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block14 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block15 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block16 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block17 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block18 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block19 = XceptionBlock([728, 728, 728, 728], dilation=middle_block_dilation, skip_connection_type='sum')
        self.block20 = XceptionBlock([728, 728, 1024, 1024], dilation=exit_block_dilations[0])
        self.block21 = XceptionBlock([1024, 1536, 1536, 2048], dilation=exit_block_dilations[1], skip_connection_type='none', relu_first=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x, low_level_feat = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)
        x = self.block21(x)
        return x, low_level_feat


def resnet(name, pretrained=False):

    def get_channels(layer):
        block = layer[-1]
        if isinstance(block, models.resnet.BasicBlock):
            return block.conv2.out_channels
        elif isinstance(block, models.resnet.Bottleneck):
            return block.conv3.out_channels
        raise RuntimeError('unknown resnet block: {}'.format(block))
    if name == 'resnet18':
        resnet = models.resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        resnet = models.resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        resnet = models.resnet50(pretrained=pretrained)
    elif name == 'resnet101':
        resnet = models.resnet101(pretrained=pretrained)
    elif name == 'resnet152':
        resnet = models.resnet152(pretrained=pretrained)
    else:
        return NotImplemented
    layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
    layer0.out_channels = resnet.bn1.num_features
    resnet.layer1.out_channels = get_channels(resnet.layer1)
    resnet.layer2.out_channels = get_channels(resnet.layer2)
    resnet.layer3.out_channels = get_channels(resnet.layer3)
    resnet.layer4.out_channels = get_channels(resnet.layer4)
    return [layer0, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]


def resnext(name, pretrained=False):
    if name in ['resnext101_32x4d', 'resnext101_64x4d']:
        pretrained = 'imagenet' if pretrained else None
        resnext = pretrainedmodels.__dict__[name](num_classes=1000, pretrained=pretrained)
    else:
        return NotImplemented
    layer0 = nn.Sequential(resnext.features[0], resnext.features[1], resnext.features[2], resnext.features[3])
    layer1 = resnext.features[4]
    layer2 = resnext.features[5]
    layer3 = resnext.features[6]
    layer4 = resnext.features[7]
    layer0.out_channels = 64
    layer1.out_channels = 256
    layer2.out_channels = 512
    layer3.out_channels = 1024
    layer4.out_channels = 2048
    return [layer0, layer1, layer2, layer3, layer4]


def se_net(name, pretrained=False):
    if name in ['se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'senet154']:
        pretrained = 'imagenet' if pretrained else None
        senet = pretrainedmodels.__dict__[name](num_classes=1000, pretrained=pretrained)
    else:
        return NotImplemented
    layer0 = senet.layer0
    layer1 = senet.layer1
    layer2 = senet.layer2
    layer3 = senet.layer3
    layer4 = senet.layer4
    layer0.out_channels = senet.layer1[0].conv1.in_channels
    layer1.out_channels = senet.layer1[-1].conv3.out_channels
    layer2.out_channels = senet.layer2[-1].conv3.out_channels
    layer3.out_channels = senet.layer3[-1].conv3.out_channels
    layer4.out_channels = senet.layer4[-1].conv3.out_channels
    return [layer0, layer1, layer2, layer3, layer4]


def create_encoder(enc_type, output_stride=8, pretrained=True):
    if enc_type.startswith('resnet'):
        return resnet(enc_type, pretrained)
    elif enc_type.startswith('resnext'):
        return resnext(enc_type, pretrained)
    elif enc_type.startswith('se'):
        return se_net(enc_type, pretrained)
    elif enc_type == 'xception65':
        return Xception65(output_stride)
    elif enc_type == 'mobilenetv2':
        return MobileNetV2(pretrained)
    else:
        raise NotImplementedError


class EncoderDecoderNet(nn.Module, SegmentatorTTA):

    def __init__(self, output_channels=19, enc_type='resnet50', dec_type='unet_scse', num_filters=16, pretrained=False):
        super().__init__()
        self.output_channels = output_channels
        self.enc_type = enc_type
        self.dec_type = dec_type
        assert enc_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext101_32x4d', 'resnext101_64x4d', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'senet154']
        assert dec_type in ['unet_scse', 'unet_seibn', 'unet_oc']
        encoder = create_encoder(enc_type, pretrained)
        Decoder = create_decoder(dec_type)
        self.encoder1 = encoder[0]
        self.encoder2 = encoder[1]
        self.encoder3 = encoder[2]
        self.encoder4 = encoder[3]
        self.encoder5 = encoder[4]
        self.pool = nn.MaxPool2d(2, 2)
        self.center = Decoder(self.encoder5.out_channels, num_filters * 32 * 2, num_filters * 32)
        self.decoder5 = Decoder(self.encoder5.out_channels + num_filters * 32, num_filters * 32 * 2, num_filters * 16)
        self.decoder4 = Decoder(self.encoder4.out_channels + num_filters * 16, num_filters * 16 * 2, num_filters * 8)
        self.decoder3 = Decoder(self.encoder3.out_channels + num_filters * 8, num_filters * 8 * 2, num_filters * 4)
        self.decoder2 = Decoder(self.encoder2.out_channels + num_filters * 4, num_filters * 4 * 2, num_filters * 2)
        self.decoder1 = Decoder(self.encoder1.out_channels + num_filters * 2, num_filters * 2 * 2, num_filters)
        self.logits = nn.Sequential(nn.Conv2d(num_filters * (16 + 8 + 4 + 2 + 1), 64, kernel_size=1, padding=0), ActivatedBatchNorm(64), nn.Conv2d(64, self.output_channels, kernel_size=1))

    def forward(self, x):
        img_size = x.shape[2:]
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        c = self.center(self.pool(e5))
        e1_up = F.interpolate(e1, scale_factor=2, mode='bilinear', align_corners=False)
        d5 = self.decoder5(c, e5)
        d4 = self.decoder4(d5, e4)
        d3 = self.decoder3(d4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1_up)
        u5 = F.interpolate(d5, img_size, mode='bilinear', align_corners=False)
        u4 = F.interpolate(d4, img_size, mode='bilinear', align_corners=False)
        u3 = F.interpolate(d3, img_size, mode='bilinear', align_corners=False)
        u2 = F.interpolate(d2, img_size, mode='bilinear', align_corners=False)
        d = torch.cat((d1, u2, u3, u4, u5), 1)
        logits = self.logits(d)
        return logits


class ASPOC(nn.Module):

    def __init__(self, in_channels=2048, out_channels=256, output_stride=8):
        super().__init__()
        if output_stride == 16:
            dilations = [6, 12, 18]
        elif output_stride == 8:
            dilations = [12, 24, 36]
        else:
            raise NotImplementedError
        self.context = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=True), ActivatedBatchNorm(out_channels), BaseOC_Context(in_channels=out_channels, out_channels=out_channels, key_channels=out_channels // 2, value_channels=out_channels, dropout=0, sizes=[2]))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=False), ActivatedBatchNorm(out_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False), ActivatedBatchNorm(out_channels))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False), ActivatedBatchNorm(out_channels))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False), ActivatedBatchNorm(out_channels))
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, padding=0, dilation=1, bias=False), ActivatedBatchNorm(out_channels), nn.Dropout2d(0.1))

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        output = self.conv_bn_dropout(out)
        return output


class ASPP(nn.Module):

    def __init__(self, in_channels=2048, out_channels=256, output_stride=8):
        super().__init__()
        if output_stride == 16:
            dilations = [6, 12, 18]
        elif output_stride == 8:
            dilations = [12, 24, 36]
        else:
            raise NotImplementedError
        self.aspp0 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False)), ('bn', nn.BatchNorm2d(out_channels)), ('relu', nn.ReLU(inplace=True))]))
        self.aspp1 = SeparableConv2d(in_channels, out_channels, dilation=dilations[0], relu_first=False)
        self.aspp2 = SeparableConv2d(in_channels, out_channels, dilation=dilations[1], relu_first=False)
        self.aspp3 = SeparableConv2d(in_channels, out_channels, dilation=dilations[2], relu_first=False)
        self.image_pooling = nn.Sequential(OrderedDict([('gap', nn.AdaptiveAvgPool2d((1, 1))), ('conv', nn.Conv2d(in_channels, out_channels, 1, bias=False)), ('bn', nn.BatchNorm2d(out_channels)), ('relu', nn.ReLU(inplace=True))]))
        self.conv = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        pool = self.image_pooling(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=True)
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x = torch.cat((pool, x0, x1, x2, x3), dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class MobileASPP(nn.Module):

    def __init__(self):
        super().__init__()
        self.aspp0 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(320, 256, 1, bias=False)), ('bn', nn.BatchNorm2d(256)), ('relu', nn.ReLU(inplace=True))]))
        self.image_pooling = nn.Sequential(OrderedDict([('gap', nn.AdaptiveAvgPool2d((1, 1))), ('conv', nn.Conv2d(320, 256, 1, bias=False)), ('bn', nn.BatchNorm2d(256)), ('relu', nn.ReLU(inplace=True))]))
        self.conv = nn.Conv2d(512, 256, 1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        pool = self.image_pooling(x)
        pool = F.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = self.aspp0(x)
        x = torch.cat((pool, x), dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SPP(nn.Module):

    def __init__(self, in_channels=2048, out_channels=256, pyramids=(1, 2, 3, 6)):
        super().__init__()
        stages = []
        for p in pyramids:
            stages.append(nn.Sequential(nn.AdaptiveAvgPool2d(p), nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), ActivatedBatchNorm(out_channels)))
        self.stages = nn.ModuleList(stages)
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels + out_channels * len(pyramids), out_channels, kernel_size=1), ActivatedBatchNorm(out_channels))

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for stage in self.stages:
            out.append(F.interpolate(stage(x), size=x_size[2:], mode='bilinear', align_corners=False))
        out = self.bottleneck(torch.cat(out, 1))
        return out


class SPPDecoder(nn.Module):

    def __init__(self, in_channels, reduced_layer_num=48):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, reduced_layer_num, 1, bias=False)
        self.bn = nn.BatchNorm2d(reduced_layer_num)
        self.relu = nn.ReLU(inplace=True)
        self.sep1 = SeparableConv2d(256 + reduced_layer_num, 256, relu_first=False)
        self.sep2 = SeparableConv2d(256, 256, relu_first=False)

    def forward(self, x, low_level_feat):
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        low_level_feat = self.conv(low_level_feat)
        low_level_feat = self.bn(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.sep1(x)
        x = self.sep2(x)
        return x


def create_mspp(dec_type):
    if dec_type == 'spp':
        return SPP(320, 256)
    elif dec_type == 'aspp':
        return ASPP(320, 256, 8)
    elif dec_type == 'oc_base':
        return BaseOC(320, 256)
    elif dec_type == 'oc_asp':
        return ASPOC(320, 256, 8)
    elif dec_type == 'maspp':
        return MobileASPP()
    elif dec_type == 'maspp_dec':
        return MobileASPP(), SPPDecoder(24, reduced_layer_num=12)
    else:
        raise NotImplementedError


def create_spp(dec_type, in_channels=2048, middle_channels=256, output_stride=8):
    if dec_type == 'spp':
        return SPP(in_channels, middle_channels), SPPDecoder(middle_channels)
    elif dec_type == 'aspp':
        return ASPP(in_channels, middle_channels, output_stride), SPPDecoder(middle_channels)
    elif dec_type == 'oc_base':
        return BaseOC(in_channels, middle_channels), SPPDecoder(middle_channels)
    elif dec_type in 'oc_asp':
        return ASPOC(in_channels, middle_channels, output_stride), SPPDecoder(middle_channels)
    else:
        raise NotImplementedError


class SPPNet(nn.Module, SegmentatorTTA):

    def __init__(self, output_channels=19, enc_type='xception65', dec_type='aspp', output_stride=8):
        super().__init__()
        self.output_channels = output_channels
        self.enc_type = enc_type
        self.dec_type = dec_type
        assert enc_type in ['xception65', 'mobilenetv2']
        assert dec_type in ['oc_base', 'oc_asp', 'spp', 'aspp', 'maspp']
        self.encoder = create_encoder(enc_type, output_stride=output_stride, pretrained=False)
        if enc_type == 'mobilenetv2':
            self.spp = create_mspp(dec_type)
        else:
            self.spp, self.decoder = create_spp(dec_type, output_stride=output_stride)
        self.logits = nn.Conv2d(256, output_channels, 1)

    def forward(self, inputs):
        if self.enc_type == 'mobilenetv2':
            x = self.encoder(inputs)
            x = self.spp(x)
            x = self.logits(x)
            return x
        else:
            x, low_level_feat = self.encoder(inputs)
            x = self.spp(x)
            x = self.decoder(x, low_level_feat)
            x = self.logits(x)
            return x

    def update_bn_eps(self):
        for m in self.encoder.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eps = 0.001

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def get_1x_lr_params(self):
        for p in self.encoder.parameters():
            yield p

    def get_10x_lr_params(self):
        modules = [self.spp, self.logits]
        if hasattr(self, 'decoder'):
            modules.append(self.decoder)
        for module in modules:
            for p in module.parameters():
                yield p


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASPOC,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2048, 64, 64])], {}),
     False),
    (ASPP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2048, 4, 4])], {}),
     True),
    (ActivatedBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BaseOC,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2048, 64, 64])], {}),
     False),
    (BaseOC_Context,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'key_channels': 4, 'value_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BinaryClassCriterion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DecoderSPP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64]), torch.rand([4, 256, 64, 64])], {}),
     True),
    (DiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (EncoderDecoderNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ExpandedConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IBN,
     lambda: ([], {'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (KlLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (LovaszLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MixedDiceBCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileASPP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 4, 4])], {}),
     True),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (OhemCrossEntropy2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (SPP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2048, 4, 4])], {}),
     False),
    (SPPDecoder,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 256, 64, 64]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SPPNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SelfAttentionBlock2D,
     lambda: ([], {'in_channels': 4, 'key_channels': 4, 'value_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SeparableConv2d,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SoftCrossEntropy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SoftIoULoss,
     lambda: ([], {'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.zeros([4, 4, 4], dtype=torch.int64)], {}),
     False),
    (Xception65,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (XceptionBlock,
     lambda: ([], {'channel_list': [4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_ActivatedBatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_nyoki_mtl_pytorch_segmentation(_paritybench_base):
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

