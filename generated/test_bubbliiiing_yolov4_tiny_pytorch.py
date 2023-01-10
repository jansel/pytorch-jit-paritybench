import sys
_module = sys.modules[__name__]
del sys
get_map = _module
kmeans_for_anchors = _module
CSPdarknet53_tiny = _module
nets = _module
attention = _module
yolo = _module
yolo_training = _module
predict = _module
summary = _module
train = _module
utils = _module
callbacks = _module
dataloader = _module
utils_bbox = _module
utils_fit = _module
utils_map = _module
coco_annotation = _module
get_map_coco = _module
voc_annotation = _module
yolo = _module

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


import math


import torch


import torch.nn as nn


from functools import partial


import numpy as np


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.optim as optim


from torch import nn


from torch.utils.data import DataLoader


import matplotlib


import scipy.signal


from matplotlib import pyplot as plt


from torch.utils.tensorboard import SummaryWriter


from random import sample


from random import shuffle


from torch.utils.data.dataset import Dataset


from torchvision.ops import nms


import time


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Resblock_body(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()
        self.out_channels = out_channels
        self.conv1 = BasicConv(in_channels, out_channels, 3)
        self.conv2 = BasicConv(out_channels // 2, out_channels // 2, 3)
        self.conv3 = BasicConv(out_channels // 2, out_channels // 2, 3)
        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        x = self.conv1(x)
        route = x
        c = self.out_channels
        x = torch.split(x, c // 2, dim=1)[1]
        x = self.conv2(x)
        route1 = x
        x = self.conv3(x)
        x = torch.cat([x, route1], dim=1)
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        x = self.maxpool(x)
        return x, feat


class CSPDarkNet(nn.Module):

    def __init__(self):
        super(CSPDarkNet, self).__init__()
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)
        self.resblock_body1 = Resblock_body(64, 64)
        self.resblock_body2 = Resblock_body(128, 128)
        self.resblock_body3 = Resblock_body(256, 256)
        self.conv3 = BasicConv(512, 512, kernel_size=3)
        self.num_features = 1
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x, _ = self.resblock_body1(x)
        x, _ = self.resblock_body2(x)
        x, feat1 = self.resblock_body3(x)
        x = self.conv3(x)
        feat2 = x
        return feat1, feat2


class se_block(nn.Module):

    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // ratio, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // ratio, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):

    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class eca_block(nn.Module):

    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CA_Block(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)
        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(BasicConv(in_channels, out_channels, 1), nn.Upsample(scale_factor=2, mode='nearest'))

    def forward(self, x):
        x = self.upsample(x)
        return x


attention_block = [se_block, cbam_block, eca_block, CA_Block]


def darknet53_tiny(pretrained, **kwargs):
    model = CSPDarkNet()
    if pretrained:
        model.load_state_dict(torch.load('model_data/CSPdarknet53_tiny_backbone_weights.pth'))
    return model


def yolo_head(filters_list, in_filters):
    m = nn.Sequential(BasicConv(in_filters, filters_list[0], 3), nn.Conv2d(filters_list[0], filters_list[1], 1))
    return m


class YoloBody(nn.Module):

    def __init__(self, anchors_mask, num_classes, phi=0, pretrained=False):
        super(YoloBody, self).__init__()
        self.phi = phi
        self.backbone = darknet53_tiny(pretrained)
        self.conv_for_P5 = BasicConv(512, 256, 1)
        self.yolo_headP5 = yolo_head([512, len(anchors_mask[0]) * (5 + num_classes)], 256)
        self.upsample = Upsample(256, 128)
        self.yolo_headP4 = yolo_head([256, len(anchors_mask[1]) * (5 + num_classes)], 384)
        if 1 <= self.phi and self.phi <= 4:
            self.feat1_att = attention_block[self.phi - 1](256)
            self.feat2_att = attention_block[self.phi - 1](512)
            self.upsample_att = attention_block[self.phi - 1](128)

    def forward(self, x):
        feat1, feat2 = self.backbone(x)
        if 1 <= self.phi and self.phi <= 4:
            feat1 = self.feat1_att(feat1)
            feat2 = self.feat2_att(feat2)
        P5 = self.conv_for_P5(feat2)
        out0 = self.yolo_headP5(P5)
        P5_Upsample = self.upsample(P5)
        if 1 <= self.phi and self.phi <= 4:
            P5_Upsample = self.upsample_att(P5_Upsample)
        P4 = torch.cat([P5_Upsample, feat1], axis=1)
        out1 = self.yolo_headP4(P4)
        return out0, out1


class YOLOLoss(nn.Module):

    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], label_smoothing=0):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.label_smoothing = label_smoothing
        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 5 * (input_shape[0] * input_shape[1]) / 416 ** 2
        self.cls_ratio = 1 * (num_classes / 80)
        self.ignore_threshold = 0.5
        self.cuda = cuda

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCELoss(self, pred, target):
        epsilon = 1e-07
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def box_ciou(self, b1, b2):
        """
        输入为：
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

        返回为：
        -------
        ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.0
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.0
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / torch.clamp(union_area, min=1e-06)
        center_distance = torch.sum(torch.pow(b1_xy - b2_xy, 2), axis=-1)
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
        ciou = iou - 1.0 * center_distance / torch.clamp(enclose_diagonal, min=1e-06)
        v = 4 / math.pi ** 2 * torch.pow(torch.atan(b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-06)) - torch.atan(b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-06)), 2)
        alpha = v / torch.clamp(1.0 - iou + v, min=1e-06)
        ciou = ciou - alpha * v
        return ciou

    def smooth_labels(self, y_true, label_smoothing, num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    def forward(self, l, input, targets=None):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)
        noobj_mask, pred_boxes = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)
        if self.cuda:
            y_true = y_true.type_as(x)
            noobj_mask = noobj_mask.type_as(x)
            box_loss_scale = box_loss_scale.type_as(x)
        box_loss_scale = 2 - box_loss_scale
        loss = 0
        obj_mask = y_true[..., 4] == 1
        n = torch.sum(obj_mask)
        if n != 0:
            ciou = self.box_ciou(pred_boxes, y_true[..., :4]).type_as(x)
            loss_loc = torch.mean((1 - ciou)[obj_mask])
            loss_cls = torch.mean(self.BCELoss(pred_cls[obj_mask], y_true[..., 5:][obj_mask]))
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio
        loss_conf = torch.mean(self.BCELoss(conf, obj_mask.type_as(conf))[noobj_mask.bool() | obj_mask])
        loss += loss_conf * self.balance[l] * self.obj_ratio
        return loss

    def calculate_iou(self, _box_a, _box_b):
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2
        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp(max_xy - min_xy, min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]
        area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
        area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
        union = area_a + area_b - inter
        return inter / union

    def get_target(self, l, targets, anchors, in_h, in_w):
        bs = len(targets)
        noobj_mask = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad=False)
        y_true = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad=False)
        for b in range(bs):
            if len(targets[b]) == 0:
                continue
            batch_target = torch.zeros_like(targets[b])
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()
            gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))
            anchor_shapes = torch.FloatTensor(torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), 1))
            iou = self.calculate_iou(gt_box, anchor_shapes)
            best_ns = torch.argmax(iou, dim=-1)
            sort_ns = torch.argsort(iou, dim=-1, descending=True)

            def check_in_anchors_mask(index, anchors_mask):
                for sub_anchors_mask in anchors_mask:
                    if index in sub_anchors_mask:
                        return True
                return False
            for t, best_n in enumerate(best_ns):
                if not check_in_anchors_mask(best_n, self.anchors_mask):
                    for index in sort_ns[t]:
                        if check_in_anchors_mask(index, self.anchors_mask):
                            best_n = index
                            break
                if best_n not in self.anchors_mask[l]:
                    continue
                k = self.anchors_mask[l].index(best_n)
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()
                c = batch_target[t, 4].long()
                noobj_mask[b, k, j, i] = 0
                y_true[b, k, j, i, 0] = batch_target[t, 0]
                y_true[b, k, j, i, 1] = batch_target[t, 1]
                y_true[b, k, j, i, 2] = batch_target[t, 2]
                y_true[b, k, j, i, 3] = batch_target[t, 3]
                y_true[b, k, j, i, 4] = 1
                y_true[b, k, j, i, c + 5] = 1
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale

    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        bs = len(targets)
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        pred_boxes_x = torch.unsqueeze(x + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y + grid_y, -1)
        pred_boxes_w = torch.unsqueeze(torch.exp(w) * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h) * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)
        for b in range(bs):
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)
            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])
                batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
                batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
                batch_target = batch_target[:, :4].type_as(x)
                anch_ious = self.calculate_iou(batch_target, pred_boxes_for_ignore)
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask, pred_boxes


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CSPDarkNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Resblock_body,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Upsample,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (eca_block,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (se_block,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_bubbliiiing_yolov4_tiny_pytorch(_paritybench_base):
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

