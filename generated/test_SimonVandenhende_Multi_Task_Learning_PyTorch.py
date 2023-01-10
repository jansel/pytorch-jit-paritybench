import sys
_module = sys.modules[__name__]
del sys
custom_transforms = _module
google_drive = _module
nyud = _module
pascal_context = _module
eval_depth = _module
eval_edge = _module
eval_human_parts = _module
eval_normals = _module
eval_sal = _module
eval_semseg = _module
evaluate_utils = _module
jaccard = _module
loss_functions = _module
loss_schemes = _module
main = _module
aspp = _module
cross_stitch = _module
layers = _module
models = _module
mtan = _module
mti_net = _module
nddr_cnn = _module
padnet = _module
resnet = _module
resnet_dilated = _module
seg_hrnet = _module
train_utils = _module
common_config = _module
config = _module
custom_collate = _module
helpers = _module
logger = _module
mypath = _module
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


import numpy.random as random


import numpy as np


import torch


import math


import torchvision


import torch.utils.data as data


import scipy.io as sio


import warnings


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.modules.module import Module


import logging


import functools


import torch._utils


import copy


from torchvision import transforms


from torch.utils.data import DataLoader


import collections


import re


from torch._six import string_classes


class SoftMaxwithLoss(Module):
    """
    This function returns cross entropy loss for semantic segmentation
    """

    def __init__(self):
        super(SoftMaxwithLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss(ignore_index=255)

    def forward(self, out, label):
        assert not label.requires_grad
        label = label[:, 0, :, :].long()
        loss = self.criterion(self.softmax(out), label)
        return loss


class BalancedCrossEntropyLoss(Module):
    """
    Balanced Cross Entropy Loss with optional ignore regions
    """

    def __init__(self, size_average=True, batch_average=True, pos_weight=None):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average
        self.pos_weight = pos_weight

    def forward(self, output, label, void_pixels=None):
        assert output.size() == label.size()
        labels = torch.ge(label, 0.5).float()
        if self.pos_weight is None:
            num_labels_pos = torch.sum(labels)
            num_labels_neg = torch.sum(1.0 - labels)
            num_total = num_labels_pos + num_labels_neg
            w = num_labels_neg / num_total
        else:
            w = self.pos_weight
        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, labels - output_gt_zero) - torch.log(1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))
        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)
        if void_pixels is not None and not self.pos_weight:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)
            num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()
            w = num_labels_neg / num_total
        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)
        final_loss = w * loss_pos + (1 - w) * loss_neg
        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]
        return final_loss


class BinaryCrossEntropyLoss(Module):
    """
    Binary Cross Entropy with ignore regions, not balanced.
    """

    def __init__(self, size_average=True, batch_average=True):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average

    def forward(self, output, label, void_pixels=None):
        assert output.size() == label.size()
        labels = torch.ge(label, 0.5).float()
        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, labels - output_gt_zero) - torch.log(1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))
        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)
        if void_pixels is not None:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)
        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)
        final_loss = loss_pos + loss_neg
        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]
        return final_loss


class DepthLoss(nn.Module):
    """
    Loss for depth prediction. By default L1 loss is used.  
    """

    def __init__(self, loss='l1'):
        super(DepthLoss, self).__init__()
        if loss == 'l1':
            self.loss = nn.L1Loss()
        else:
            raise NotImplementedError('Loss {} currently not supported in DepthLoss'.format(loss))

    def forward(self, out, label):
        mask = label != 255
        return self.loss(torch.masked_select(out, mask), torch.masked_select(label, mask))


class Normalize(nn.Module):

    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)
        return top


class NormalsLoss(Module):
    """
    L1 loss with ignore labels
    normalize: normalization for surface normals
    """

    def __init__(self, size_average=True, normalize=False, norm=1):
        super(NormalsLoss, self).__init__()
        self.size_average = size_average
        if normalize:
            self.normalize = Normalize()
        else:
            self.normalize = None
        if norm == 1:
            None
            self.loss_func = F.l1_loss
        elif norm == 2:
            None
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError

    def forward(self, out, label, ignore_label=255):
        assert not label.requires_grad
        mask = label != ignore_label
        n_valid = torch.sum(mask).item()
        if self.normalize is not None:
            out_norm = self.normalize(out)
            loss = self.loss_func(torch.masked_select(out_norm, mask), torch.masked_select(label, mask), reduction='sum')
        else:
            loss = self.loss_func(torch.masked_select(out, mask), torch.masked_select(label, mask), reduction='sum')
        if self.size_average:
            if ignore_label:
                ret_loss = torch.div(loss, max(n_valid, 1e-06))
                return ret_loss
            else:
                ret_loss = torch.div(loss, float(np.prod(label.size())))
                return ret_loss
        return loss


class SingleTaskLoss(nn.Module):

    def __init__(self, loss_ft, task):
        super(SingleTaskLoss, self).__init__()
        self.loss_ft = loss_ft
        self.task = task

    def forward(self, pred, gt):
        out = {self.task: self.loss_ft(pred[self.task], gt[self.task])}
        out['total'] = out[self.task]
        return out


class MultiTaskLoss(nn.Module):

    def __init__(self, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss, self).__init__()
        assert set(tasks) == set(loss_ft.keys())
        assert set(tasks) == set(loss_weights.keys())
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
        out['total'] = torch.sum(torch.stack([(self.loss_weights[t] * out[t]) for t in self.tasks]))
        return out


class PADNetLoss(nn.Module):

    def __init__(self, tasks: list, auxilary_tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(PADNetLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt):
        total = 0.0
        out = {}
        img_size = gt[self.tasks[0]].size()[-2:]
        for task in self.auxilary_tasks:
            pred_ = F.interpolate(pred['initial_%s' % task], img_size, mode='bilinear')
            gt_ = gt[task]
            loss_ = self.loss_ft[task](pred_, gt_)
            out['deepsup_%s' % task] = loss_
            total += self.loss_weights[task] * loss_
        for task in self.tasks:
            pred_, gt_ = pred[task], gt[task]
            loss_ = self.loss_ft[task](pred_, gt_)
            out[task] = loss_
            total += self.loss_weights[task] * loss_
        out['total'] = total
        return out


class MTINetLoss(nn.Module):

    def __init__(self, tasks: list, auxilary_tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MTINetLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt):
        total = 0.0
        out = {}
        img_size = gt[self.tasks[0]].size()[-2:]
        for scale in range(4):
            pred_scale = pred['deep_supervision']['scale_%s' % scale]
            pred_scale = {t: F.interpolate(pred_scale[t], img_size, mode='bilinear') for t in self.auxilary_tasks}
            losses_scale = {t: self.loss_ft[t](pred_scale[t], gt[t]) for t in self.auxilary_tasks}
            for k, v in losses_scale.items():
                out['scale_%d_%s' % (scale, k)] = v
                total += self.loss_weights[k] * v
        losses_out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
        for k, v in losses_out.items():
            out[k] = v
            total += self.loss_weights[k] * v
        out['total'] = total
        return out


class ASPPConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation):
        modules = [nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU()]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):

    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabHead(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(ASPP(in_channels, [12, 24, 36]), nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, num_classes, 1))


class ChannelWiseMultiply(nn.Module):

    def __init__(self, num_channels):
        super(ChannelWiseMultiply, self).__init__()
        self.param = nn.Parameter(torch.FloatTensor(num_channels), requires_grad=True)

    def init_value(self, value):
        with torch.no_grad():
            self.param.data.fill_(value)

    def forward(self, x):
        return torch.mul(self.param.view(1, -1, 1, 1), x)


class CrossStitchUnit(nn.Module):

    def __init__(self, tasks, num_channels, alpha, beta):
        super(CrossStitchUnit, self).__init__()
        self.cross_stitch_unit = nn.ModuleDict({t: nn.ModuleDict({t: ChannelWiseMultiply(num_channels) for t in tasks}) for t in tasks})
        for t_i in tasks:
            for t_j in tasks:
                if t_i == t_j:
                    self.cross_stitch_unit[t_i][t_j].init_value(alpha)
                else:
                    self.cross_stitch_unit[t_i][t_j].init_value(beta)

    def forward(self, task_features):
        out = {}
        for t_i in task_features.keys():
            prod = torch.stack([self.cross_stitch_unit[t_i][t_j](task_features[t_j]) for t_j in task_features.keys()])
            out[t_i] = torch.sum(prod, dim=0)
        return out


class CrossStitchNetwork(nn.Module):
    """ 
        Implementation of cross-stitch networks.
        We insert a cross-stitch unit, to combine features from the task-specific backbones
        after every stage.
       
        Argument: 
            backbone: 
                nn.ModuleDict object which contains pre-trained task-specific backbones.
                {task: backbone for task in p.TASKS.NAMES}
        
            heads: 
                nn.ModuleDict object which contains the task-specific heads.
                {task: head for task in p.TASKS.NAMES}
        
            stages: 
                list of stages where we instert a cross-stitch unit between the task-specific backbones.
                Note: the backbone modules require a method 'forward_stage' to get feature representations
                at the respective stages.
        
            channels: 
                dict which contains the number of channels in every stage
        
            alpha, beta: 
                floats for initializing cross-stitch units (see paper)
        
    """

    def __init__(self, p, backbone: nn.ModuleDict, heads: nn.ModuleDict, stages: list, channels: dict, alpha: float, beta: float):
        super(CrossStitchNetwork, self).__init__()
        self.tasks = p.TASKS.NAMES
        self.backbone = backbone
        self.heads = heads
        self.stages = stages
        self.cross_stitch = nn.ModuleDict({stage: CrossStitchUnit(self.tasks, channels[stage], alpha, beta) for stage in stages})

    def forward(self, x):
        img_size = x.size()[-2:]
        x = {task: x for task in self.tasks}
        for stage in self.stages:
            for task in self.tasks:
                x[task] = self.backbone[task].forward_stage(x[task], stage)
            x = self.cross_stitch[stage](x)
        out = {task: self.heads[task](x[task]) for task in self.tasks}
        out = {task: F.interpolate(out[task], img_size, mode='bilinear') for task in self.tasks}
        return out


class SEBlock(nn.Module):
    """ Squeeze-and-excitation block """

    def __init__(self, channels, r=16):
        super(SEBlock, self).__init__()
        self.r = r
        self.squeeze = nn.Sequential(nn.Linear(channels, channels // self.r), nn.ReLU(), nn.Linear(channels // self.r, channels), nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.size()
        squeeze = self.squeeze(torch.mean(x, dim=(2, 3))).view(B, C, 1, 1)
        return torch.mul(x, squeeze)


class SABlock(nn.Module):
    """ Spatial self-attention block """

    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        attention_mask = self.attention(x)
        features = self.conv(x)
        return torch.mul(features, attention_mask)


class SingleTaskModel(nn.Module):
    """ Single-task baseline model with encoder + decoder """

    def __init__(self, backbone: nn.Module, decoder: nn.Module, task: str):
        super(SingleTaskModel, self).__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.task = task

    def forward(self, x):
        out_size = x.size()[2:]
        out = self.decoder(self.backbone(x))
        return {self.task: F.interpolate(out, out_size, mode='bilinear')}


class MultiTaskModel(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """

    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list):
        super(MultiTaskModel, self).__init__()
        assert set(decoders.keys()) == set(tasks)
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks

    def forward(self, x):
        out_size = x.size()[2:]
        shared_representation = self.backbone(x)
        return {task: F.interpolate(self.decoders[task](shared_representation), out_size, mode='bilinear') for task in self.tasks}


class AttentionLayer(nn.Sequential):
    """ 
        Attention layer: Takes a feature representation as input and generates an attention mask 
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(AttentionLayer, self).__init__(nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, padding=0), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True), nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, padding=0), nn.BatchNorm2d(out_channels), nn.Sigmoid())


BN_MOMENTUM = 0.01


BatchNorm2d = functools.partial(nn.BatchNorm2d)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
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
        out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class RefinementBlock(nn.Sequential):
    """
        Refinement block uses a single Bottleneck layer to refine the features after applying task-specific attention.
    """

    def __init__(self, in_channels, out_channels):
        downsample = nn.Sequential(conv1x1(in_channels, out_channels, stride=1), nn.BatchNorm2d(out_channels))
        super(RefinementBlock, self).__init__(Bottleneck(in_channels, out_channels // 4, downsample=downsample))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResnetDilated(nn.Module):
    """ ResNet backbone with dilated convolutions """

    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial
        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = 1, 1
                if m.kernel_size == (3, 3):
                    m.dilation = dilate // 2, dilate // 2
                    m.padding = dilate // 2, dilate // 2
            elif m.kernel_size == (3, 3):
                m.dilation = dilate, dilate
                m.padding = dilate, dilate

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_stage(self, x, stage):
        assert stage in ['conv', 'layer1', 'layer2', 'layer3', 'layer4', 'layer1_without_conv']
        if stage == 'conv':
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            return x
        elif stage == 'layer1':
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            return x
        elif stage == 'layer1_without_conv':
            x = self.layer1(x)
            return x
        else:
            layer = getattr(self, stage)
            return layer(x)

    def forward_stage_except_last_block(self, x, stage):
        assert stage in ['layer1', 'layer2', 'layer3', 'layer4']
        if stage == 'layer1':
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1[:-1](x)
            return x
        else:
            layer = getattr(self, stage)
            return layer[:-1](x)

    def forward_stage_last_block(self, x, stage):
        assert stage in ['layer1', 'layer2', 'layer3', 'layer4']
        if stage == 'layer1':
            x = self.layer1[-1](x)
            return x
        else:
            layer = getattr(self, stage)
            return layer[-1](x)


class MTAN(nn.Module):
    """ 
        Implementation of MTAN  
        https://arxiv.org/abs/1803.10704 

        Note: The implementation is based on a ResNet backbone.
        
        Argument: 
            backbone: 
                nn.ModuleDict object which contains pre-trained task-specific backbones.
                {task: backbone for task in p.TASKS.NAMES}
        
            heads: 
                nn.ModuleDict object which contains the task-specific heads.
                {task: head for task in p.TASKS.NAMES}
        
            stages: 
                a list of the different stages in the network 
                ['layer1', 'layer2', 'layer3', 'layer4']
 
            channels: 
                dict which contains the number of channels in every stage
            
            downsample:
                dict which tells where to apply 2 x 2 downsampling in the model        

    """

    def __init__(self, p, backbone, heads: nn.ModuleDict, stages: list, channels: dict, downsample: dict):
        super(MTAN, self).__init__()
        self.tasks = p.TASKS.NAMES
        assert isinstance(backbone, ResNet) or isinstance(backbone, ResnetDilated)
        self.backbone = backbone
        self.heads = heads
        assert set(stages) == {'layer1', 'layer2', 'layer3', 'layer4'}
        self.stages = stages
        self.channels = channels
        self.attention_1 = nn.ModuleDict({task: AttentionLayer(channels['layer1'], channels['layer1'] // 4, channels['layer1']) for task in self.tasks})
        self.attention_2 = nn.ModuleDict({task: AttentionLayer(2 * channels['layer2'], channels['layer2'] // 4, channels['layer2']) for task in self.tasks})
        self.attention_3 = nn.ModuleDict({task: AttentionLayer(2 * channels['layer3'], channels['layer3'] // 4, channels['layer3']) for task in self.tasks})
        self.attention_4 = nn.ModuleDict({task: AttentionLayer(2 * channels['layer4'], channels['layer4'] // 4, channels['layer4']) for task in self.tasks})
        self.refine_1 = RefinementBlock(channels['layer1'], channels['layer2'])
        self.refine_2 = RefinementBlock(channels['layer2'], channels['layer3'])
        self.refine_3 = RefinementBlock(channels['layer3'], channels['layer4'])
        self.downsample = {stage: (nn.MaxPool2d(kernel_size=2, stride=2) if downsample else nn.Identity()) for stage, downsample in downsample.items()}

    def forward(self, x):
        img_size = x.size()[-2:]
        u_1_b = self.backbone.forward_stage_except_last_block(x, 'layer1')
        u_1_t = self.backbone.forward_stage_last_block(u_1_b, 'layer1')
        u_2_b = self.backbone.forward_stage_except_last_block(u_1_t, 'layer2')
        u_2_t = self.backbone.forward_stage_last_block(u_2_b, 'layer2')
        u_3_b = self.backbone.forward_stage_except_last_block(u_2_t, 'layer3')
        u_3_t = self.backbone.forward_stage_last_block(u_3_b, 'layer3')
        u_4_b = self.backbone.forward_stage_except_last_block(u_3_t, 'layer4')
        u_4_t = self.backbone.forward_stage_last_block(u_4_b, 'layer4')
        a_1_mask = {task: self.attention_1[task](u_1_b) for task in self.tasks}
        a_1 = {task: (a_1_mask[task] * u_1_t) for task in self.tasks}
        a_1 = {task: self.downsample['layer1'](self.refine_1(a_1[task])) for task in self.tasks}
        a_2_mask = {task: self.attention_2[task](torch.cat((u_2_b, a_1[task]), 1)) for task in self.tasks}
        a_2 = {task: (a_2_mask[task] * u_2_t) for task in self.tasks}
        a_2 = {task: self.downsample['layer2'](self.refine_2(a_2[task])) for task in self.tasks}
        a_3_mask = {task: self.attention_3[task](torch.cat((u_3_b, a_2[task]), 1)) for task in self.tasks}
        a_3 = {task: (a_3_mask[task] * u_3_t) for task in self.tasks}
        a_3 = {task: self.downsample['layer3'](self.refine_3(a_3[task])) for task in self.tasks}
        a_4_mask = {task: self.attention_4[task](torch.cat((u_4_b, a_3[task]), 1)) for task in self.tasks}
        a_4 = {task: (a_4_mask[task] * u_4_t) for task in self.tasks}
        out = {task: self.heads[task](a_4[task]) for task in self.tasks}
        out = {task: F.interpolate(out[task], img_size, mode='bilinear') for task in self.tasks}
        return out


class InitialTaskPredictionModule(nn.Module):
    """
        Make the initial task predictions from the backbone features.
    """

    def __init__(self, p, tasks, input_channels, intermediate_channels=256):
        super(InitialTaskPredictionModule, self).__init__()
        self.tasks = tasks
        layers = {}
        conv_out = {}
        for task in self.tasks:
            if input_channels != intermediate_channels:
                downsample = nn.Sequential(nn.Conv2d(input_channels, intermediate_channels, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(intermediate_channels))
            else:
                downsample = None
            bottleneck1 = Bottleneck(input_channels, intermediate_channels // 4, downsample=downsample)
            bottleneck2 = Bottleneck(intermediate_channels, intermediate_channels // 4, downsample=None)
            conv_out_ = nn.Conv2d(intermediate_channels, p.AUXILARY_TASKS.NUM_OUTPUT[task], 1)
            layers[task] = nn.Sequential(bottleneck1, bottleneck2)
            conv_out[task] = conv_out_
        self.layers = nn.ModuleDict(layers)
        self.conv_out = nn.ModuleDict(conv_out)

    def forward(self, x):
        out = {}
        for task in self.tasks:
            out['features_%s' % task] = self.layers[task](x)
            out[task] = self.conv_out[task](out['features_%s' % task])
        return out


class FPM(nn.Module):
    """ Feature Propagation Module """

    def __init__(self, auxilary_tasks, per_task_channels):
        super(FPM, self).__init__()
        self.auxilary_tasks = auxilary_tasks
        self.N = len(self.auxilary_tasks)
        self.per_task_channels = per_task_channels
        self.shared_channels = int(self.N * per_task_channels)
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.shared_channels // 4, 1, bias=False), nn.BatchNorm2d(self.shared_channels // 4))
        self.non_linear = nn.Sequential(BasicBlock(self.shared_channels, self.shared_channels // 4, downsample=downsample), BasicBlock(self.shared_channels // 4, self.shared_channels // 4), nn.Conv2d(self.shared_channels // 4, self.shared_channels, 1))
        downsample = nn.Sequential(nn.Conv2d(self.shared_channels, self.per_task_channels, 1, bias=False), nn.BatchNorm2d(self.per_task_channels))
        self.dimensionality_reduction = BasicBlock(self.shared_channels, self.per_task_channels, downsample=downsample)
        self.se = nn.ModuleDict({task: SEBlock(self.per_task_channels) for task in self.auxilary_tasks})

    def forward(self, x):
        concat = torch.cat([x['features_%s' % task] for task in self.auxilary_tasks], 1)
        B, C, H, W = concat.size()
        shared = self.non_linear(concat)
        mask = F.softmax(shared.view(B, C // self.N, self.N, H, W), dim=2)
        shared = torch.mul(mask, concat.view(B, C // self.N, self.N, H, W)).view(B, -1, H, W)
        shared = self.dimensionality_reduction(shared)
        out = {}
        for task in self.auxilary_tasks:
            out[task] = self.se[task](shared) + x['features_%s' % task]
        return out


class MultiTaskDistillationModule(nn.Module):
    """
        Perform Multi-Task Distillation
        We apply an attention mask to features from other tasks and
        add the result as a residual.
    """

    def __init__(self, tasks, auxilary_tasks, channels):
        super(MultiTaskDistillationModule, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.self_attention = {}
        for t in self.tasks:
            other_tasks = [a for a in self.auxilary_tasks if a != t]
            self.self_attention[t] = nn.ModuleDict({a: SABlock(channels, channels) for a in other_tasks})
        self.self_attention = nn.ModuleDict(self.self_attention)

    def forward(self, x):
        adapters = {t: {a: self.self_attention[t][a](x['features_%s' % a]) for a in self.auxilary_tasks if a != t} for t in self.tasks}
        out = {t: (x['features_%s' % t] + torch.sum(torch.stack([v for v in adapters[t].values()]), dim=0)) for t in self.tasks}
        return out


class MTINet(nn.Module):
    """ 
        MTI-Net implementation based on HRNet backbone 
        https://arxiv.org/pdf/2001.06902.pdf
    """

    def __init__(self, p, backbone, backbone_channels, heads):
        super(MTINet, self).__init__()
        self.tasks = p.TASKS.NAMES
        self.auxilary_tasks = p.AUXILARY_TASKS.NAMES
        self.num_scales = len(backbone_channels)
        self.channels = backbone_channels
        self.backbone = backbone
        self.fpm_scale_3 = FPM(self.auxilary_tasks, self.channels[3])
        self.fpm_scale_2 = FPM(self.auxilary_tasks, self.channels[2])
        self.fpm_scale_1 = FPM(self.auxilary_tasks, self.channels[1])
        self.scale_0 = InitialTaskPredictionModule(p, self.auxilary_tasks, self.channels[0] + self.channels[1], self.channels[0])
        self.scale_1 = InitialTaskPredictionModule(p, self.auxilary_tasks, self.channels[1] + self.channels[2], self.channels[1])
        self.scale_2 = InitialTaskPredictionModule(p, self.auxilary_tasks, self.channels[2] + self.channels[3], self.channels[2])
        self.scale_3 = InitialTaskPredictionModule(p, self.auxilary_tasks, self.channels[3], self.channels[3])
        self.distillation_scale_0 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[0])
        self.distillation_scale_1 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[1])
        self.distillation_scale_2 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[2])
        self.distillation_scale_3 = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, self.channels[3])
        self.heads = heads

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}
        x = self.backbone(x)
        x_3 = self.scale_3(x[3])
        x_3_fpm = self.fpm_scale_3(x_3)
        x_2 = self.scale_2(x[2], x_3_fpm)
        x_2_fpm = self.fpm_scale_2(x_2)
        x_1 = self.scale_1(x[1], x_2_fpm)
        x_1_fpm = self.fpm_scale_1(x_1)
        x_0 = self.scale_0(x[0], x_1_fpm)
        out['deep_supervision'] = {'scale_0': x_0, 'scale_1': x_1, 'scale_2': x_2, 'scale_3': x_3}
        features_0 = self.distillation_scale_0(x_0)
        features_1 = self.distillation_scale_1(x_1)
        features_2 = self.distillation_scale_2(x_2)
        features_3 = self.distillation_scale_3(x_3)
        multi_scale_features = {t: [features_0[t], features_1[t], features_2[t], features_3[t]] for t in self.tasks}
        for t in self.tasks:
            out[t] = F.interpolate(self.heads[t](multi_scale_features[t]), img_size, mode='bilinear')
        return out


class NDDRLayer(nn.Module):

    def __init__(self, tasks, channels, alpha, beta):
        super(NDDRLayer, self).__init__()
        self.tasks = tasks
        self.layer = nn.ModuleDict({task: nn.Sequential(nn.Conv2d(len(tasks) * channels, channels, 1, 1, 0, bias=False), nn.BatchNorm2d(channels, momentum=0.05), nn.ReLU()) for task in self.tasks})
        for i, task in enumerate(self.tasks):
            layer = self.layer[task]
            t_alpha = torch.diag(torch.FloatTensor([alpha for _ in range(channels)]))
            t_beta = torch.diag(torch.FloatTensor([beta for _ in range(channels)])).repeat(1, len(self.tasks))
            t_alpha = t_alpha.view(channels, channels, 1, 1)
            t_beta = t_beta.view(channels, channels * len(self.tasks), 1, 1)
            layer[0].weight.data.copy_(t_beta)
            layer[0].weight.data[:, int(i * channels):int((i + 1) * channels)].copy_(t_alpha)
            layer[1].weight.data.fill_(1.0)
            layer[1].bias.data.fill_(0.0)

    def forward(self, x):
        x = torch.cat([x[task] for task in self.tasks], 1)
        output = {task: self.layer[task](x) for task in self.tasks}
        return output


class NDDRCNN(nn.Module):
    """ 
        Implementation of NDDR-CNN.
        We insert a nddr-layer to fuse the features from the task-specific backbones after every
        stage.
       
        Argument: 
            backbone: 
                nn.ModuleDict object which contains pre-trained task-specific backbones.
                {task: backbone for task in p.TASKS.NAMES}
        
            heads: 
                nn.ModuleDict object which contains the task-specific heads.
                {task: head for task in p.TASKS.NAMES}

            all_stages:        

            nddr_stages: 
                list of stages where we instert a nddr-layer between the task-specific backbones.
                Note: the backbone modules require a method 'forward_stage' to get feature representations
                at the respective stages.
        
            channels: 
                dict which contains the number of channels in every stage
        
            alpha, beta: 
                floats for initializing the nddr-layers (see paper)
        
    """

    def __init__(self, p, backbone: nn.ModuleDict, heads: nn.ModuleDict, all_stages: list, nddr_stages: list, channels: dict, alpha: float, beta: float):
        super(NDDRCNN, self).__init__()
        self.tasks = p.TASKS.NAMES
        self.backbone = backbone
        self.heads = heads
        self.all_stages = all_stages
        self.nddr_stages = nddr_stages
        self.nddr = nn.ModuleDict({stage: NDDRLayer(self.tasks, channels[stage], alpha, beta) for stage in nddr_stages})

    def forward(self, x):
        img_size = x.size()[-2:]
        x = {task: x for task in self.tasks}
        for stage in self.all_stages:
            for task in self.tasks:
                x[task] = self.backbone[task].forward_stage(x[task], stage)
            if stage in self.nddr_stages:
                x = self.nddr[stage](x)
        out = {task: self.heads[task](x[task]) for task in self.tasks}
        out = {task: F.interpolate(out[task], img_size, mode='bilinear') for task in self.tasks}
        return out


class PADNet(nn.Module):

    def __init__(self, p, backbone, backbone_channels):
        super(PADNet, self).__init__()
        self.tasks = p.TASKS.NAMES
        self.auxilary_tasks = p.AUXILARY_TASKS.NAMES
        self.channels = backbone_channels
        self.backbone = backbone
        self.initial_task_prediction_heads = InitialTaskPredictionModule(p, self.auxilary_tasks, self.channels)
        self.multi_modal_distillation = MultiTaskDistillationModule(self.tasks, self.auxilary_tasks, 256)
        heads = {}
        for task in self.tasks:
            bottleneck1 = Bottleneck(256, 256 // 4, downsample=None)
            bottleneck2 = Bottleneck(256, 256 // 4, downsample=None)
            conv_out_ = nn.Conv2d(256, p.AUXILARY_TASKS.NUM_OUTPUT[task], 1)
            heads[task] = nn.Sequential(bottleneck1, bottleneck2, conv_out_)
        self.heads = nn.ModuleDict(heads)

    def forward(self, x):
        img_size = x.size()[-2:]
        out = {}
        x = self.backbone(x)
        x = self.initial_task_prediction_heads(x)
        for task in self.auxilary_tasks:
            out['initial_%s' % task] = x[task]
        x = self.multi_modal_distillation(x)
        for task in self.tasks:
            out[task] = F.interpolate(self.heads[task](x[task]), img_size, mode='bilinear')
        return out


logger = logging.getLogger(__name__)


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False), BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM), nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=[height_output, width_output], mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        extra = config['MODEL']['EXTRA']
        super(HighResolutionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels
        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)
        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)
        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)
        last_inp_channels = np.int(np.sum(pre_stage_channels))

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False), BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM), nn.ReLU(inplace=False)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False), BatchNorm2d(outchannels, momentum=BN_MOMENTUM), nn.ReLU(inplace=False)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        return x

    def init_weights(self, pretrained=''):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            None
            pretrained_dict = torch.load(pretrained)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


class HighResolutionFuse(nn.Module):

    def __init__(self, backbone_channels, num_outputs):
        super(HighResolutionFuse, self).__init__()
        last_inp_channels = sum(backbone_channels)
        self.last_layer = nn.Sequential(nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(last_inp_channels, momentum=0.1), nn.ReLU(inplace=False))

    def forward(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')
        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        return x


class HighResolutionHead(nn.Module):

    def __init__(self, backbone_channels, num_outputs):
        super(HighResolutionHead, self).__init__()
        last_inp_channels = sum(backbone_channels)
        self.last_layer = nn.Sequential(nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(last_inp_channels, momentum=0.1), nn.ReLU(inplace=False), nn.Conv2d(in_channels=last_inp_channels, out_channels=num_outputs, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], (x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], (x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], (x0_h, x0_w), mode='bilinear')
        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASPP,
     lambda: ([], {'in_channels': 4, 'atrous_rates': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ASPPConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ASPPPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (AttentionLayer,
     lambda: ([], {'in_channels': 4, 'mid_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BalancedCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BinaryCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChannelWiseMultiply,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DeepLabHead,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DepthLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Normalize,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NormalsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (RefinementBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SABlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SEBlock,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SingleTaskLoss,
     lambda: ([], {'loss_ft': MSELoss(), 'task': 4}),
     lambda: ([torch.rand([5, 4, 4, 4]), torch.rand([5, 4, 4, 4])], {}),
     False),
    (SingleTaskModel,
     lambda: ([], {'backbone': _mock_layer(), 'decoder': _mock_layer(), 'task': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SoftMaxwithLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_SimonVandenhende_Multi_Task_Learning_PyTorch(_paritybench_base):
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

