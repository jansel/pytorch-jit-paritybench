import sys
_module = sys.modules[__name__]
del sys
cfg = _module
darknet = _module
voc_label = _module
dataset = _module
debug = _module
demo = _module
detect = _module
eval = _module
focal_loss = _module
image = _module
bn = _module
bn_lib = _module
build = _module
caffe_net = _module
resnet = _module
tiny_yolo = _module
outputs = _module
partial = _module
recall = _module
region_layer = _module
coco_eval = _module
eval_all = _module
eval_ap = _module
eval_widerface = _module
my_eval = _module
create_dataset = _module
lmdb_utils = _module
plot_lmdb = _module
train_lmdb = _module
train = _module
utils = _module
valid = _module
yolo_layer = _module

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
xrange = range
wraps = functools.wraps


import torch


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import random


from torch.utils.data import Dataset


import torch.optim as optim


import time


from torchvision import datasets


from torchvision import transforms


import math


from torch.autograd import Variable


from torch.nn.parameter import Parameter


from torch.autograd import Function


from collections import OrderedDict


import torch.backends.cudnn as cudnn


import itertools


import types


class MaxPoolStride1(nn.Module):

    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode='replicate'), 2, stride=1)
        return x


class Upsample(nn.Module):

    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert x.data.dim() == 4
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, hs, W, ws).contiguous().view(B, C, H * hs, W * ws)
        return x


class Reorg(nn.Module):

    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert x.data.dim() == 4
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert H % stride == 0
        assert W % stride == 0
        ws = stride
        hs = stride
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, H // hs * (W // ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H // hs, W // ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H // hs, W // ws)
        return x


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x


class EmptyModule(nn.Module):

    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = min(box1[0], box2[0])
        x2_max = max(box1[2], box2[2])
        y1_min = min(box1[1], box2[1])
        y2_max = max(box1[3], box2[3])
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    else:
        w1, h1 = box1[2], box1[3]
        w2, h2 = box2[2], box2[3]
        x1_min = min(box1[0] - w1 / 2.0, box2[0] - w2 / 2.0)
        x2_max = max(box1[0] + w1 / 2.0, box2[0] + w2 / 2.0)
        y1_min = min(box1[1] - h1 / 2.0, box2[1] - h2 / 2.0)
        y2_max = max(box1[1] + h1 / 2.0, box2[1] + h2 / 2.0)
    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    carea = 0
    if w_cross <= 0 or h_cross <= 0:
        return 0.0
    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    uarea = area1 + area2 - carea
    return float(carea / uarea)


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def multi_bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = torch.min(boxes1[0], boxes2[0])
        x2_max = torch.max(boxes1[2], boxes2[2])
        y1_min = torch.min(boxes1[1], boxes2[1])
        y2_max = torch.max(boxes1[3], boxes2[3])
        w1, h1 = boxes1[2] - boxes1[0], boxes1[3] - boxes1[1]
        w2, h2 = boxes2[2] - boxes2[0], boxes2[3] - boxes2[1]
    else:
        w1, h1 = boxes1[2], boxes1[3]
        w2, h2 = boxes2[2], boxes2[3]
        x1_min = torch.min(boxes1[0] - w1 / 2.0, boxes2[0] - w2 / 2.0)
        x2_max = torch.max(boxes1[0] + w1 / 2.0, boxes2[0] + w2 / 2.0)
        y1_min = torch.min(boxes1[1] - h1 / 2.0, boxes2[1] - h2 / 2.0)
        y2_max = torch.max(boxes1[1] + h1 / 2.0, boxes2[1] + h2 / 2.0)
    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    mask = (w_cross <= 0) + (h_cross <= 0) > 0
    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


class RegionLayer(nn.Module):

    def __init__(self, num_classes=0, anchors=[1.0], num_anchors=1, use_cuda=None):
        super(RegionLayer, self).__init__()
        use_cuda = torch.cuda.is_available() and (True if use_cuda is None else use_cuda)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.anchors = torch.FloatTensor(anchors).view(self.num_anchors, self.anchor_step)
        self.rescore = 1
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

    def build_targets(self, pred_boxes, target, nH, nW):
        nB = target.size(0)
        nA = self.num_anchors
        noobj_mask = torch.ones(nB, nA, nH, nW)
        obj_mask = torch.zeros(nB, nA, nH, nW)
        coord_mask = torch.zeros(nB, nA, nH, nW)
        tcoord = torch.zeros(4, nB, nA, nH, nW)
        tconf = torch.zeros(nB, nA, nH, nW)
        tcls = torch.zeros(nB, nA, nH, nW)
        nAnchors = nA * nH * nW
        nPixels = nH * nW
        nGT = 0
        nRecall = 0
        anchors = self.anchors
        if self.seen < 12800:
            tcoord[0].fill_(0.5)
            tcoord[1].fill_(0.5)
            coord_mask.fill_(0.01)
        for b in range(nB):
            cur_pred_boxes = pred_boxes[b * nAnchors:(b + 1) * nAnchors].t()
            cur_ious = torch.zeros(nAnchors)
            tbox = target[b].view(-1, 5)
            for t in range(50):
                if tbox[t][1] == 0:
                    break
                gx, gw = [(i * nW) for i in (tbox[t][1], tbox[t][3])]
                gy, gh = [(i * nH) for i in (tbox[t][2], tbox[t][4])]
                cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors, 1).t()
                cur_ious = torch.max(cur_ious, multi_bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
            ignore_ix = (cur_ious > self.thresh).view(nA, nH, nW)
            noobj_mask[b][ignore_ix] = 0
            for t in range(50):
                if tbox[t][1] == 0:
                    break
                nGT += 1
                gx, gw = [(i * nW) for i in (tbox[t][1], tbox[t][3])]
                gy, gh = [(i * nH) for i in (tbox[t][2], tbox[t][4])]
                gw, gh = gw.float(), gh.float()
                gi, gj = int(gx), int(gy)
                tmp_gt_boxes = torch.FloatTensor([0, 0, gw, gh]).repeat(nA, 1).t()
                anchor_boxes = torch.cat((torch.zeros(nA, 2), anchors), 1).t()
                tmp_ious = multi_bbox_ious(anchor_boxes, tmp_gt_boxes, x1y1x2y2=False)
                best_iou, best_n = torch.max(tmp_ious, 0)
                if self.anchor_step == 4:
                    tmp_ious_mask = tmp_ious == best_iou
                    if tmp_ious_mask.sum() > 0:
                        gt_pos = torch.FloatTensor([gi, gj, gx, gy]).repeat(nA, 1).t()
                        an_pos = anchor_boxes[4:6]
                        dist = pow(gt_pos[0] + an_pos[0] - gt_pos[2], 2) + pow(gt_pos[1] + an_pos[1] - gt_pos[3], 2)
                        dist[1 - tmp_ious_mask] = 10000
                        _, best_n = torch.min(dist, 0)
                gt_box = torch.FloatTensor([gx, gy, gw, gh])
                pred_box = pred_boxes[b * nAnchors + best_n * nPixels + gj * nW + gi]
                iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
                obj_mask[b][best_n][gj][gi] = 1
                noobj_mask[b][best_n][gj][gi] = 0
                coord_mask[b][best_n][gj][gi] = 2.0 - tbox[t][3] * tbox[t][4]
                tcoord[0][b][best_n][gj][gi] = gx - gi
                tcoord[1][b][best_n][gj][gi] = gy - gj
                tcoord[2][b][best_n][gj][gi] = math.log(gw / anchors[best_n][0])
                tcoord[3][b][best_n][gj][gi] = math.log(gh / anchors[best_n][1])
                tcls[b][best_n][gj][gi] = tbox[t][0]
                tconf[b][best_n][gj][gi] = iou if self.rescore else 1.0
                if iou > 0.5:
                    nRecall += 1
        return nGT, nRecall, obj_mask, noobj_mask, coord_mask, tcoord, tconf, tcls

    def get_mask_boxes(self, output):
        if not isinstance(self.anchors, torch.Tensor):
            self.anchors = torch.FloatTensor(self.anchors).view(self.num_anchors, self.anchor_step)
        masked_anchors = self.anchors.view(-1)
        num_anchors = torch.IntTensor([self.num_anchors])
        return {'x': output, 'a': masked_anchors, 'n': num_anchors}

    def forward(self, output, target):
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        cls_anchor_dim = nB * nA * nH * nW
        if not isinstance(self.anchors, torch.Tensor):
            self.anchors = torch.FloatTensor(self.anchors).view(self.num_anchors, self.anchor_step)
        output = output.view(nB, nA, 5 + nC, nH, nW)
        cls_grid = torch.linspace(5, 5 + nC - 1, nC).long()
        ix = torch.LongTensor(range(0, 5))
        pred_boxes = torch.FloatTensor(4, cls_anchor_dim)
        coord = output.index_select(2, ix[0:4]).view(nB * nA, -1, nH * nW).transpose(0, 1).contiguous().view(-1, cls_anchor_dim)
        coord[0:2] = coord[0:2].sigmoid()
        conf = output.index_select(2, ix[4]).view(cls_anchor_dim).sigmoid()
        cls = output.index_select(2, cls_grid)
        cls = cls.view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(cls_anchor_dim, nC)
        t1 = time.time()
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nB * nA, nH, 1).view(cls_anchor_dim)
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(cls_anchor_dim)
        anchor_w = self.anchors.index_select(1, ix[0]).repeat(nB, nH * nW).view(cls_anchor_dim)
        anchor_h = self.anchors.index_select(1, ix[1]).repeat(nB, nH * nW).view(cls_anchor_dim)
        pred_boxes[0] = coord[0] + grid_x
        pred_boxes[1] = coord[1] + grid_y
        pred_boxes[2] = coord[2].exp() * anchor_w
        pred_boxes[3] = coord[3].exp() * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 4)).detach()
        t2 = time.time()
        nGT, nRecall, obj_mask, noobj_mask, coord_mask, tcoord, tconf, tcls = self.build_targets(pred_boxes, target.detach(), nH, nW)
        cls_mask = obj_mask == 1
        tcls = tcls[cls_mask].long().view(-1)
        cls_mask = cls_mask.view(-1, 1).repeat(1, nC)
        cls = cls[cls_mask].view(-1, nC)
        nProposals = int((conf > 0.25).sum())
        tcoord = tcoord.view(4, cls_anchor_dim)
        tconf = tconf.view(cls_anchor_dim)
        conf_mask = (self.object_scale * obj_mask + self.noobject_scale * noobj_mask).view(cls_anchor_dim)
        obj_mask = obj_mask.view(cls_anchor_dim)
        coord_mask = coord_mask.view(cls_anchor_dim)
        t3 = time.time()
        loss_coord = self.coord_scale * nn.MSELoss(reduction='sum')(coord * coord_mask, tcoord * coord_mask) / nB
        loss_conf = nn.MSELoss(reduction='sum')(conf * conf_mask, tconf * conf_mask) / nB
        loss_cls = self.class_scale * nn.CrossEntropyLoss(reduction='sum')(cls, tcls) / nB
        loss = loss_coord + loss_conf + loss_cls
        t4 = time.time()
        if False:
            None
            None
            None
            None
            None
            None
        None
        if math.isnan(loss.item()):
            None
            sys.exit(0)
        return loss


class YoloLayer(nn.Module):

    def __init__(self, anchor_mask=[], num_classes=0, anchors=[1.0], num_anchors=1, use_cuda=None):
        super(YoloLayer, self).__init__()
        use_cuda = torch.cuda.is_available() and (True if use_cuda is None else use_cuda)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.rescore = 1
        self.ignore_thresh = 0.5
        self.truth_thresh = 1.0
        self.nth_layer = 0
        self.seen = 0
        self.net_width = 0
        self.net_height = 0

    def get_mask_boxes(self, output):
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
        masked_anchors = torch.FloatTensor(masked_anchors)
        num_anchors = torch.IntTensor([len(self.anchor_mask)])
        return {'x': output, 'a': masked_anchors, 'n': num_anchors}

    def build_targets(self, pred_boxes, target, anchors, nA, nH, nW):
        nB = target.size(0)
        anchor_step = anchors.size(1)
        noobj_mask = torch.ones(nB, nA, nH, nW)
        obj_mask = torch.zeros(nB, nA, nH, nW)
        coord_mask = torch.zeros(nB, nA, nH, nW)
        tcoord = torch.zeros(4, nB, nA, nH, nW)
        tconf = torch.zeros(nB, nA, nH, nW)
        tcls = torch.zeros(nB, nA, nH, nW, self.num_classes)
        nAnchors = nA * nH * nW
        nPixels = nH * nW
        nGT = 0
        nRecall = 0
        nRecall75 = 0
        anchors = anchors
        for b in range(nB):
            cur_pred_boxes = pred_boxes[b * nAnchors:(b + 1) * nAnchors].t()
            cur_ious = torch.zeros(nAnchors)
            tbox = target[b].view(-1, 5)
            for t in range(50):
                if tbox[t][1] == 0:
                    break
                gx, gy = tbox[t][1] * nW, tbox[t][2] * nH
                gw, gh = tbox[t][3] * self.net_width, tbox[t][4] * self.net_height
                cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors, 1).t()
                cur_ious = torch.max(cur_ious, multi_bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
            ignore_ix = (cur_ious > self.ignore_thresh).view(nA, nH, nW)
            noobj_mask[b][ignore_ix] = 0
            for t in range(50):
                if tbox[t][1] == 0:
                    break
                nGT += 1
                gx, gy = tbox[t][1] * nW, tbox[t][2] * nH
                gw, gh = tbox[t][3] * self.net_width, tbox[t][4] * self.net_height
                gw, gh = gw.float(), gh.float()
                gi, gj = int(gx), int(gy)
                tmp_gt_boxes = torch.FloatTensor([0, 0, gw, gh]).repeat(nA, 1).t()
                anchor_boxes = torch.cat((torch.zeros(nA, anchor_step), anchors), 1).t()
                _, best_n = torch.max(multi_bbox_ious(anchor_boxes, tmp_gt_boxes, x1y1x2y2=False), 0)
                gt_box = torch.FloatTensor([gx, gy, gw, gh])
                pred_box = pred_boxes[b * nAnchors + best_n * nPixels + gj * nW + gi]
                iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
                obj_mask[b][best_n][gj][gi] = 1
                noobj_mask[b][best_n][gj][gi] = 0
                coord_mask[b][best_n][gj][gi] = 2.0 - tbox[t][3] * tbox[t][4]
                tcoord[0][b][best_n][gj][gi] = gx - gi
                tcoord[1][b][best_n][gj][gi] = gy - gj
                tcoord[2][b][best_n][gj][gi] = math.log(gw / anchors[best_n][0])
                tcoord[3][b][best_n][gj][gi] = math.log(gh / anchors[best_n][1])
                tcls[b][best_n][gj][gi][int(tbox[t][0])] = 1
                tconf[b][best_n][gj][gi] = iou if self.rescore else 1.0
                if iou > 0.5:
                    nRecall += 1
                    if iou > 0.75:
                        nRecall75 += 1
        return nGT, nRecall, nRecall75, obj_mask, noobj_mask, coord_mask, tcoord, tconf, tcls

    def forward(self, output, target):
        mask_tuple = self.get_mask_boxes(output)
        t0 = time.time()
        nB = output.data.size(0)
        nA = mask_tuple['n'].item()
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        anchor_step = mask_tuple['a'].size(0) // nA
        anchors = mask_tuple['a'].view(nA, anchor_step)
        cls_anchor_dim = nB * nA * nH * nW
        output = output.view(nB, nA, 5 + nC, nH, nW)
        cls_grid = torch.linspace(5, 5 + nC - 1, nC).long()
        ix = torch.LongTensor(range(0, 5))
        pred_boxes = torch.FloatTensor(4, cls_anchor_dim)
        coord = output.index_select(2, ix[0:4]).view(nB * nA, -1, nH * nW).transpose(0, 1).contiguous().view(-1, cls_anchor_dim)
        coord[0:2] = coord[0:2].sigmoid()
        conf = output.index_select(2, ix[4]).view(cls_anchor_dim).sigmoid()
        cls = output.index_select(2, cls_grid)
        cls = cls.view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(cls_anchor_dim, nC)
        t1 = time.time()
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nB * nA, nH, 1).view(cls_anchor_dim)
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(cls_anchor_dim)
        anchor_w = anchors.index_select(1, ix[0]).repeat(nB, nH * nW).view(cls_anchor_dim)
        anchor_h = anchors.index_select(1, ix[1]).repeat(nB, nH * nW).view(cls_anchor_dim)
        pred_boxes[0] = coord[0] + grid_x
        pred_boxes[1] = coord[1] + grid_y
        pred_boxes[2] = coord[2].exp() * anchor_w
        pred_boxes[3] = coord[3].exp() * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 4)).detach()
        t2 = time.time()
        nGT, nRecall, nRecall75, obj_mask, noobj_mask, coord_mask, tcoord, tconf, tcls = self.build_targets(pred_boxes, target.detach(), anchors.detach(), nA, nH, nW)
        conf_mask = (obj_mask + noobj_mask).view(cls_anchor_dim)
        obj_mask = (obj_mask == 1).view(cls_anchor_dim)
        nProposals = int((conf > 0.25).sum())
        coord = coord[:, (obj_mask)]
        tcoord = tcoord.view(4, cls_anchor_dim)[:, (obj_mask)]
        tconf = tconf.view(cls_anchor_dim)
        cls = cls[(obj_mask), :]
        tcls = tcls.view(cls_anchor_dim, nC)[(obj_mask), :]
        t3 = time.time()
        loss_coord = nn.BCELoss(reduction='sum')(coord[0:2], tcoord[0:2]) / nB + nn.MSELoss(reduction='sum')(coord[2:4], tcoord[2:4]) / nB
        loss_conf = nn.BCELoss(reduction='sum')(conf * conf_mask, tconf * conf_mask) / nB
        loss_cls = nn.BCEWithLogitsLoss(reduction='sum')(cls, tcls) / nB
        loss = loss_coord + loss_conf + loss_cls
        t4 = time.time()
        if False:
            None
            None
            None
            None
            None
            None
        None
        if math.isnan(loss.item()):
            None
            sys.exit(0)
        return loss


def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]).view_as(conv_model.bias.data))
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).view_as(conv_model.weight.data))
    start = start + num_w
    return start


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).view_as(conv_model.weight.data))
    start = start + num_w
    return start


def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]))
    start = start + num_w
    return start


def parse_cfg(cfgfile):
    blocks = []
    fp = open(cfgfile, 'r')
    block = None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key, value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()
    if block:
        blocks.append(block)
    fp.close()
    return blocks


def print_cfg(blocks):
    None
    prev_width = 416
    prev_height = 416
    prev_filters = 3
    out_filters = []
    out_widths = []
    out_heights = []
    ind = -2
    for block in blocks:
        ind = ind + 1
        if block['type'] == 'net':
            prev_width = int(block['width'])
            prev_height = int(block['height'])
            continue
        elif block['type'] == 'convolutional':
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = (kernel_size - 1) // 2 if is_pad else 0
            width = (prev_width + 2 * pad - kernel_size) // stride + 1
            height = (prev_height + 2 * pad - kernel_size) // stride + 1
            None
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            width = prev_width // stride
            height = prev_height // stride
            None
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'avgpool':
            width = 1
            height = 1
            None
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'softmax':
            None
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'cost':
            None
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'reorg':
            stride = int(block['stride'])
            filters = stride * stride * prev_filters
            width = prev_width // stride
            height = prev_height // stride
            None
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            filters = prev_filters
            width = prev_width * stride
            height = prev_height * stride
            None
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [(int(i) if int(i) > 0 else int(i) + ind) for i in layers]
            if len(layers) == 1:
                None
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                None
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert prev_width == out_widths[layers[1]]
                assert prev_height == out_heights[layers[1]]
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] in ['region', 'yolo']:
            None
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'shortcut':
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id + ind
            None
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'connected':
            filters = int(block['output'])
            None
            prev_filters = filters
            out_widths.append(1)
            out_heights.append(1)
            out_filters.append(prev_filters)
        else:
            None


def save_conv(fp, conv_model):
    if conv_model.bias.is_cuda:
        convert2cpu(conv_model.bias.data).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        conv_model.bias.data.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)


def save_conv_bn(fp, conv_model, bn_model):
    if bn_model.bias.is_cuda:
        convert2cpu(bn_model.bias.data).numpy().tofile(fp)
        convert2cpu(bn_model.weight.data).numpy().tofile(fp)
        convert2cpu(bn_model.running_mean).numpy().tofile(fp)
        convert2cpu(bn_model.running_var).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        bn_model.bias.data.numpy().tofile(fp)
        bn_model.weight.data.numpy().tofile(fp)
        bn_model.running_mean.numpy().tofile(fp)
        bn_model.running_var.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)


def save_fc(fp, fc_model):
    fc_model.bias.data.numpy().tofile(fp)
    fc_model.weight.data.numpy().tofile(fp)


class Darknet(nn.Module):

    def net_name(self):
        names_list = 'region', 'yolo'
        name = names_list[0]
        for m in self.models:
            if isinstance(m, YoloLayer):
                name = names_list[1]
        return name

    def getLossLayers(self):
        loss_layers = []
        for m in self.models:
            if isinstance(m, RegionLayer) or isinstance(m, YoloLayer):
                loss_layers.append(m)
        return loss_layers

    def __init__(self, cfgfile, use_cuda=True):
        super(Darknet, self).__init__()
        self.use_cuda = use_cuda
        self.blocks = parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks)
        self.loss_layers = self.getLossLayers()
        if len(self.loss_layers) > 0:
            last = len(self.loss_layers) - 1
            self.anchors = self.loss_layers[last].anchors
            self.num_anchors = self.loss_layers[last].num_anchors
            self.anchor_step = self.loss_layers[last].anchor_step
            self.num_classes = self.loss_layers[last].num_classes
        self.header = torch.IntTensor([0, 1, 0, 0])
        self.seen = 0

    def forward(self, x):
        ind = -2
        outputs = dict()
        out_boxes = dict()
        outno = 0
        for block in self.blocks:
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'maxpool', 'reorg', 'upsample', 'avgpool', 'softmax', 'connected']:
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [(int(i) if int(i) > 0 else int(i) + ind) for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1, x2), 1)
                outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind - 1]
                x = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] in ['region', 'yolo']:
                boxes = self.models[ind].get_mask_boxes(x)
                out_boxes[outno] = boxes
                outno += 1
                outputs[ind] = None
            elif block['type'] == 'cost':
                continue
            else:
                None
        return x if outno == 0 else out_boxes

    def print_network(self):
        print_cfg(self.blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()
        prev_filters = 3
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0
        ind = -2
        for block in blocks:
            ind += 1
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                self.width = int(block['width'])
                self.height = int(block['height'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size - 1) // 2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_strides.append(prev_stride)
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                prev_stride = prev_stride * stride
                out_strides.append(prev_stride)
                models.append(Reorg(stride))
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                out_filters.append(prev_filters)
                prev_stride = prev_stride / stride
                out_strides.append(prev_stride)
                models.append(Upsample(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [(int(i) if int(i) > 0 else int(i) + ind) for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                    prev_stride = out_strides[layers[0]]
                elif len(layers) == 2:
                    assert layers[0] == ind - 1
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                    prev_stride = out_strides[layers[0]]
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind - 1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(nn.Linear(prev_filters, filters), nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(nn.Linear(prev_filters, filters), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'region':
                region_layer = RegionLayer(use_cuda=self.use_cuda)
                anchors = block['anchors'].split(',')
                region_layer.anchors = [float(i) for i in anchors]
                region_layer.num_classes = int(block['classes'])
                region_layer.num_anchors = int(block['num'])
                region_layer.anchor_step = len(region_layer.anchors) // region_layer.num_anchors
                region_layer.rescore = int(block['rescore'])
                region_layer.object_scale = float(block['object_scale'])
                region_layer.noobject_scale = float(block['noobject_scale'])
                region_layer.class_scale = float(block['class_scale'])
                region_layer.coord_scale = float(block['coord_scale'])
                region_layer.thresh = float(block['thresh'])
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(region_layer)
            elif block['type'] == 'yolo':
                yolo_layer = YoloLayer(use_cuda=self.use_cuda)
                anchors = block['anchors'].split(',')
                anchor_mask = block['mask'].split(',')
                yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
                yolo_layer.anchors = [float(i) for i in anchors]
                yolo_layer.num_classes = int(block['classes'])
                yolo_layer.num_anchors = int(block['num'])
                yolo_layer.anchor_step = len(yolo_layer.anchors) // yolo_layer.num_anchors
                try:
                    yolo_layer.rescore = int(block['rescore'])
                except:
                    pass
                yolo_layer.ignore_thresh = float(block['ignore_thresh'])
                yolo_layer.truth_thresh = float(block['truth_thresh'])
                yolo_layer.stride = prev_stride
                yolo_layer.nth_layer = ind
                yolo_layer.net_width = self.width
                yolo_layer.net_height = self.height
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(yolo_layer)
            else:
                None
        return models

    def load_binfile(self, weightfile):
        fp = open(weightfile, 'rb')
        version = np.fromfile(fp, count=3, dtype=np.int32)
        version = [int(i) for i in version]
        if version[0] * 10 + version[1] >= 2 and version[0] < 1000 and version[1] < 1000:
            seen = np.fromfile(fp, count=1, dtype=np.int64)
        else:
            seen = np.fromfile(fp, count=1, dtype=np.int32)
        self.header = torch.from_numpy(np.concatenate((version, seen), axis=0))
        self.seen = int(seen)
        body = np.fromfile(fp, dtype=np.float32)
        fp.close()
        return body

    def load_weights(self, weightfile):
        buf = self.load_binfile(weightfile)
        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'yolo':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                None

    def save_weights(self, outfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks) - 1
        dirname = os.path.dirname(outfile)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = np.array(self.header[0:3].numpy(), np.int32)
        header.tofile(fp)
        if self.header[0] * 10 + self.header[1] >= 2:
            seen = np.array(self.seen, np.int64)
        else:
            seen = np.array(self.seen, np.int32)
        seen.tofile(fp)
        ind = -1
        for blockId in range(1, cutoff + 1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    save_fc(fp, model)
                else:
                    save_fc(fp, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'yolo':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                None
        fp.close()


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            
            Loss(x, class) = - \\alpha (1-softmax(x)[class])^gamma \\log(softmax(x)[class])
    
        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.

    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        None
        C = inputs.size(1)
        P = F.softmax(inputs)
        class_mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.0)
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -alpha * torch.pow(1 - probs, self.gamma) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class BN2dFunc(Function):

    def __init__(self, running_mean, running_var, training, momentum, eps):
        self.running_mean = running_mean
        self.running_var = running_var
        self.training = training
        self.momentum = momentum
        self.eps = eps

    def forward(self, input, weight, bias):
        nB = input.size(0)
        nC = input.size(1)
        nH = input.size(2)
        nW = input.size(3)
        output = input.new(nB, nC, nH, nW)
        self.input = input
        self.weight = weight
        self.bias = bias
        self.x = input.new(nB, nC, nH, nW)
        self.x_norm = input.new(nB, nC, nH, nW)
        self.mean = input.new(nB, nC)
        self.var = input.new(nB, nC)
        if input.is_cuda:
            bn_lib.bn_forward_gpu(input, self.x, self.x_norm, self.mean, self.running_mean, self.var, self.running_var, weight, bias, self.training, output)
        else:
            bn_lib.bn_forward(input, self.x, self.x_norm, self.mean, self.running_mean, self.var, self.running_var, weight, bias, self.training, output)
        return output

    def backward(self, grad_output):
        nB = grad_output.size(0)
        nC = grad_output.size(1)
        nH = grad_output.size(2)
        nW = grad_output.size(3)
        grad_input = grad_output.new(nB, nC, nH, nW)
        grad_mean = grad_output.new(nC)
        grad_var = grad_output.new(nC)
        grad_weight = grad_output.new(nC)
        grad_bias = grad_output.new(nC)
        if grad_output.is_cuda:
            bn_lib.bn_backward_gpu(grad_output, self.input, self.x_norm, self.mean, grad_mean, self.var, grad_var, self.weight, grad_weight, self.bias, grad_bias, self.training, grad_input)
        else:
            bn_lib.bn_backward(grad_output, self.input, self.x_norm, self.mean, grad_mean, self.var, grad_var, self.weight, grad_weight, self.bias, grad_bias, self.training, grad_input)
        return grad_input, grad_weight, grad_bias


class BN2d(nn.Module):

    def __init__(self, num_features, momentum=0.01, eps=1e-05):
        super(BN2d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.momentum = momentum
        self.eps = eps
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, input):
        return BN2dFunc(self.running_mean, self.running_var, self.training, self.momentum, self.eps)(input, self.weight, self.bias)


class BN2d_slow(nn.Module):

    def __init__(self, num_features, momentum=0.01):
        super(BN2d_slow, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.eps = 1e-05
        self.momentum = momentum
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, x):
        nB = x.data.size(0)
        nC = x.data.size(1)
        nH = x.data.size(2)
        nW = x.data.size(3)
        samples = nB * nH * nW
        y = x.view(nB, nC, nH * nW).transpose(1, 2).contiguous().view(-1, nC)
        if self.training:
            None
            m = Variable(y.mean(0).data, requires_grad=False)
            v = Variable(y.var(0).data, requires_grad=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * m.data.view(-1)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * v.data.view(-1)
            m = m.repeat(samples, 1)
            v = v.repeat(samples, 1) * (samples - 1.0) / samples
        else:
            m = Variable(self.running_mean.repeat(samples, 1), requires_grad=False)
            v = Variable(self.running_var.repeat(samples, 1), requires_grad=False)
        w = self.weight.repeat(samples, 1)
        b = self.bias.repeat(samples, 1)
        y = (y - m) / (v + self.eps).sqrt() * w + b
        y = y.view(nB, nH * nW, nC).transpose(1, 2).contiguous().view(nB, nC, nH, nW)
        return y


class Scale(nn.Module):

    def __init__(self):
        super(Scale, self).__init__()

    def forward(self, x):
        return x


class Eltwise(nn.Module):

    def __init__(self, operation='+'):
        super(Eltwise, self).__init__()
        self.operation = operation

    def forward(self, input_feats):
        if isinstance(input_feats, tuple):
            None
        for i, feat in enumerate(input_feats):
            if x is None:
                x = feat
                continue
            if self.operation == '+':
                x += feat
            if self.operation == '*':
                x *= feat
            if self.operation == '/':
                x /= feat
        return x


class Concat(nn.Module):

    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, input_feats):
        if not isinstance(input_feats, tuple):
            None
        self.length = len(input_feats)
        x = torch.cat(input_feats, 1)
        return x


def parse_prototxt(protofile):

    def line_type(line):
        if line.find(':') >= 0:
            return 0
        elif line.find('{') >= 0:
            return 1
        return -1

    def parse_param_block(fp):
        block = dict()
        line = fp.readline().strip()
        while line != '}':
            ltype = line_type(line)
            if ltype == 0:
                key, value = line.split(':')
                key = key.strip()
                value = value.strip().strip('"')
                block[key] = value
            elif ltype == 1:
                key = line.split('{')[0].strip()
                sub_block = parse_param_block(fp)
                block[key] = sub_block
            line = fp.readline().strip()
        return block

    def parse_layer_block(fp):
        block = dict()
        block['top'] = []
        block['bottom'] = []
        line = fp.readline().strip()
        while line != '}':
            ltype = line_type(line)
            if ltype == 0:
                key, value = line.split(':')
                key = key.strip()
                value = value.strip().strip('"')
                if key == 'top' or key == 'bottom':
                    block[key].append(value)
                else:
                    block[key] = value
            elif ltype == 1:
                key = line.split('{')[0].strip()
                sub_block = parse_param_block(fp)
                block[key] = sub_block
            line = fp.readline().strip()
        return block
    fp = open(protofile, 'r')
    props = dict()
    layers = []
    line = fp.readline()
    while line != '':
        ltype = line_type(line)
        if ltype == 0:
            key, value = line.split(':')
            key = key.strip()
            value = value.strip().strip('"')
            props[key] = value
        elif ltype == 1:
            key = line.split('{')[0].strip()
            assert key == 'layer' or key == 'input_shape'
            layer = parse_layer_block(fp)
            layers.append(layer)
        line = fp.readline()
    net_info = dict()
    net_info['props'] = props
    net_info['layers'] = layers
    return net_info


class CaffeNet(nn.Module):

    def __init__(self, protofile, caffemodel):
        super(CaffeNet, self).__init__()
        self.seen = 0
        self.num_classes = 1
        self.is_pretrained = True
        if not caffemodel is None:
            self.is_pretrained = True
        self.anchors = [0.625, 0.75, 0.625, 0.75, 0.625, 0.75, 0.625, 0.75, 0.625, 0.75, 1.0, 1.2, 1.0, 1.2, 1.0, 1.2, 1.0, 1.2, 1.6, 1.92, 2.56, 3.072, 4.096, 4.915, 6.554, 7.864, 10.486, 12.583]
        self.num_anchors = len(self.anchors) / 2
        self.width = 480
        self.height = 320
        self.loss = RegionLoss(self.num_classes, self.anchors, self.num_anchors)
        self.net_info = parse_prototxt(protofile)
        self.models = self.create_network(self.net_info)
        self.modelList = nn.ModuleList()
        if self.is_pretrained:
            self.load_weigths_from_caffe(protofile, caffemodel)
        for name, model in list(self.models.items()):
            self.modelList.append(model)

    def load_weigths_from_caffe(self, protofile, caffemodel):
        caffe.set_mode_cpu()
        net = caffe.Net(protofile, caffemodel, caffe.TEST)
        for name, layer in list(self.models.items()):
            if isinstance(layer, nn.Conv2d):
                caffe_weight = net.params[name][0].data
                layer.weight.data = torch.from_numpy(caffe_weight)
                if len(net.params[name]) > 1:
                    caffe_bias = net.params[name][1].data
                    layer.bias.data = torch.from_numpy(caffe_bias)
                continue
            if isinstance(layer, nn.BatchNorm2d):
                caffe_means = net.params[name][0].data
                caffe_var = net.params[name][1].data
                layer.running_mean = torch.from_numpy(caffe_means)
                layer.running_var = torch.from_numpy(caffe_var)
                top_name_of_bn = self.layer_map_to_top[name][0]
                scale_name = ''
                for caffe_layer in self.net_info['layers']:
                    if caffe_layer['type'] == 'Scale' and caffe_layer['bottom'][0] == top_name_of_bn:
                        scale_name = caffe_layer['name']
                        break
                if scale_name != '':
                    caffe_weight = net.params[scale_name][0].data
                    layer.weight.data = torch.from_numpy(caffe_weight)
                    if len(net.params[name]) > 1:
                        caffe_bias = net.params[scale_name][1].data
                        layer.bias.data = torch.from_numpy(caffe_bias)

    def print_network(self):
        None

    def create_network(self, net_info):
        models = OrderedDict()
        top_dim = {'data': 3}
        self.layer_map_to_bottom = dict()
        self.layer_map_to_top = dict()
        for layer in net_info['layers']:
            name = layer['name']
            ltype = layer['type']
            if ltype == 'Data':
                continue
            if ltype == 'ImageData':
                continue
            if 'top' in layer:
                tops = layer['top']
                self.layer_map_to_top[name] = tops
            if 'bottom' in layer:
                bottoms = layer['bottom']
                self.layer_map_to_bottom[name] = bottoms
            if ltype == 'Convolution':
                filters = int(layer['convolution_param']['num_output'])
                kernel_size = int(layer['convolution_param']['kernel_size'])
                stride = 1
                group = 1
                pad = 0
                bias = True
                dilation = 1
                if 'stride' in layer['convolution_param']:
                    stride = int(layer['convolution_param']['stride'])
                if 'pad' in layer['convolution_param']:
                    pad = int(layer['convolution_param']['pad'])
                if 'group' in layer['convolution_param']:
                    group = int(layer['convolution_param']['group'])
                if 'bias_term' in layer['convolution_param']:
                    bias = True if layer['convolution_param']['bias_term'].lower() == 'false' else False
                if 'dilation' in layer['convolution_param']:
                    dilation = int(layer['convolution_param']['dilation'])
                num_output = int(layer['convolution_param']['num_output'])
                top_dim[tops[0]] = num_output
                num_input = top_dim[bottoms[0]]
                models[name] = nn.Conv2d(num_input, num_output, kernel_size, stride, pad, groups=group, bias=bias, dilation=dilation)
            elif ltype == 'ReLU':
                inplace = bottoms == tops
                top_dim[tops[0]] = top_dim[bottoms[0]]
                models[name] = nn.ReLU(inplace=False)
            elif ltype == 'Pooling':
                kernel_size = int(layer['pooling_param']['kernel_size'])
                stride = 1
                if 'stride' in layer['pooling_param']:
                    stride = int(layer['pooling_param']['stride'])
                top_dim[tops[0]] = top_dim[bottoms[0]]
                models[name] = nn.MaxPool2d(kernel_size, stride)
            elif ltype == 'BatchNorm':
                if 'use_global_stats' in layer['batch_norm_param']:
                    use_global_stats = True if layer['batch_norm_param']['use_global_stats'].lower() == 'true' else False
                top_dim[tops[0]] = top_dim[bottoms[0]]
                models[name] = nn.BatchNorm2d(top_dim[bottoms[0]])
            elif ltype == 'Scale':
                top_dim[tops[0]] = top_dim[bottoms[0]]
                models[name] = Scale()
            elif ltype == 'Eltwise':
                top_dim[tops[0]] = top_dim[bottoms[0]]
                models[name] = Eltwise('+')
            elif ltype == 'Concat':
                top_dim[tops[0]] = 0
                for i, x in enumerate(bottoms):
                    top_dim[tops[0]] += top_dim[x]
                models[name] = Concat()
            elif ltype == 'Dropout':
                if layer['top'][0] == layer['bottom'][0]:
                    inplace = True
                else:
                    inplace = False
                top_dim[tops[0]] = top_dim[bottoms[0]]
                models[name] = nn.Dropout2d(inplace=inplace)
            else:
                None
        return models

    def forward(self, x, target=None):
        blobs = OrderedDict()
        for name, layer in list(self.models.items()):
            output_names = self.layer_map_to_top[name]
            input_names = self.layer_map_to_bottom[name]
            None
            None
            None
            None
            if input_names[0] == 'data':
                top_blobs = layer(x)
            else:
                input_blobs = [blobs[i] for i in input_names]
                if isinstance(layer, Concat) or isinstance(layer, Eltwise):
                    top_blobs = layer(input_blobs)
                else:
                    top_blobs = layer(input_blobs[0])
            if not isinstance(top_blobs, tuple):
                top_blobs = top_blobs,
            for k, v in zip(output_names, top_blobs):
                blobs[k] = v
        output_name = list(blobs.keys())[-1]
        None
        return blobs[output_name]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
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
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Resnet101(nn.Module):

    def __init__(self):
        super(Resnet, self).__init__()
        self.seen = 0
        self.num_classes = 20
        self.anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
        self.num_anchors = len(self.anchors) / 2
        num_output = (5 + self.num_classes) * self.num_anchors
        self.width = 160
        self.height = 160
        self.loss = RegionLoss(self.num_classes, self.anchors, self.num_anchors)
        self.model = ResNet(Bottleneck, [3, 4, 6, 3])

    def forward(self, x):
        x = self.model(x)
        return x

    def print_network(self):
        None


class TinyYoloNet(nn.Module):

    def __init__(self):
        super(TinyYoloNet, self).__init__()
        self.seen = 0
        self.num_classes = 20
        self.anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
        self.num_anchors = len(self.anchors) / 2
        num_output = (5 + self.num_classes) * self.num_anchors
        self.width = 160
        self.height = 160
        self.loss = RegionLoss(self.num_classes, self.anchors, self.num_anchors)
        self.cnn = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 16, 3, 1, 1, bias=False)), ('bn1', nn.BatchNorm2d(16)), ('leaky1', nn.LeakyReLU(0.1, inplace=True)), ('pool1', nn.MaxPool2d(2, 2)), ('conv2', nn.Conv2d(16, 32, 3, 1, 1, bias=False)), ('bn2', nn.BatchNorm2d(32)), ('leaky2', nn.LeakyReLU(0.1, inplace=True)), ('pool2', nn.MaxPool2d(2, 2)), ('conv3', nn.Conv2d(32, 64, 3, 1, 1, bias=False)), ('bn3', nn.BatchNorm2d(64)), ('leaky3', nn.LeakyReLU(0.1, inplace=True)), ('pool3', nn.MaxPool2d(2, 2)), ('conv4', nn.Conv2d(64, 128, 3, 1, 1, bias=False)), ('bn4', nn.BatchNorm2d(128)), ('leaky4', nn.LeakyReLU(0.1, inplace=True)), ('pool4', nn.MaxPool2d(2, 2)), ('conv5', nn.Conv2d(128, 256, 3, 1, 1, bias=False)), ('bn5', nn.BatchNorm2d(256)), ('leaky5', nn.LeakyReLU(0.1, inplace=True)), ('pool5', nn.MaxPool2d(2, 2)), ('conv6', nn.Conv2d(256, 512, 3, 1, 1, bias=False)), ('bn6', nn.BatchNorm2d(512)), ('leaky6', nn.LeakyReLU(0.1, inplace=True)), ('pool6', MaxPoolStride1()), ('conv7', nn.Conv2d(512, 1024, 3, 1, 1, bias=False)), ('bn7', nn.BatchNorm2d(1024)), ('leaky7', nn.LeakyReLU(0.1, inplace=True)), ('conv8', nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)), ('bn8', nn.BatchNorm2d(1024)), ('leaky8', nn.LeakyReLU(0.1, inplace=True)), ('output', nn.Conv2d(1024, num_output, 1, 1, 0))]))

    def forward(self, x):
        x = self.cnn(x)
        return x

    def print_network(self):
        None

    def load_weights(self, path):
        buf = np.fromfile(path, dtype=np.float32)
        start = 4
        start = load_conv_bn(buf, start, self.cnn[0], self.cnn[1])
        start = load_conv_bn(buf, start, self.cnn[4], self.cnn[5])
        start = load_conv_bn(buf, start, self.cnn[8], self.cnn[9])
        start = load_conv_bn(buf, start, self.cnn[12], self.cnn[13])
        start = load_conv_bn(buf, start, self.cnn[16], self.cnn[17])
        start = load_conv_bn(buf, start, self.cnn[20], self.cnn[21])
        start = load_conv_bn(buf, start, self.cnn[24], self.cnn[25])
        start = load_conv_bn(buf, start, self.cnn[27], self.cnn[28])
        start = load_conv(buf, start, self.cnn[30])


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BN2d_slow,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Darknet,
     lambda: ([], {'cfgfile': _mock_config()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EmptyModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxPoolStride1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Reorg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Scale,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Upsample,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_andy_yun_pytorch_0_4_yolov3(_paritybench_base):
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

