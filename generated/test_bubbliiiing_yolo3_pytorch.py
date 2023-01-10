import sys
_module = sys.modules[__name__]
del sys
get_map = _module
kmeans_for_anchors = _module
nets = _module
darknet = _module
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


from collections import OrderedDict


import torch.nn as nn


import torch


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


from torch.utils.data.dataset import Dataset


from torchvision.ops import nms


import time


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out += residual
        return out


class DarkNet(nn.Module):

    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)
        self.layer1 = self._make_layer([32, 64], layers[0])
        self.layer2 = self._make_layer([64, 128], layers[1])
        self.layer3 = self._make_layer([128, 256], layers[2])
        self.layer4 = self._make_layer([256, 512], layers[3])
        self.layer5 = self._make_layer([512, 1024], layers[4])
        self.layers_out_filters = [64, 128, 256, 512, 1024]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        layers = []
        layers.append(('ds_conv', nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(('ds_bn', nn.BatchNorm2d(planes[1])))
        layers.append(('ds_relu', nn.LeakyReLU(0.1)))
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(('residual_{}'.format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        return out3, out4, out5


def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([('conv', nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)), ('bn', nn.BatchNorm2d(filter_out)), ('relu', nn.LeakyReLU(0.1))]))


def darknet53():
    model = DarkNet([1, 2, 8, 8, 4])
    return model


def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.Sequential(conv2d(in_filters, filters_list[0], 1), conv2d(filters_list[0], filters_list[1], 3), conv2d(filters_list[1], filters_list[0], 1), conv2d(filters_list[0], filters_list[1], 3), conv2d(filters_list[1], filters_list[0], 1), conv2d(filters_list[0], filters_list[1], 3), nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True))
    return m


class YoloBody(nn.Module):

    def __init__(self, anchors_mask, num_classes, pretrained=False):
        super(YoloBody, self).__init__()
        self.backbone = darknet53()
        if pretrained:
            self.backbone.load_state_dict(torch.load('model_data/darknet53_backbone_weights.pth'))
        out_filters = self.backbone.layers_out_filters
        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))
        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))
        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)
        out0_branch = self.last_layer0[:5](x0)
        out0 = self.last_layer0[5:](out0_branch)
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1_branch = self.last_layer1[:5](x1_in)
        out1 = self.last_layer1[5:](out1_branch)
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2 = self.last_layer2(x2_in)
        return out0, out1, out2


class YOLOLoss(nn.Module):

    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.giou = True
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

    def box_giou(self, b1, b2):
        """
        输入为：
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

        返回为：
        -------
        giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
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
        iou = intersect_area / union_area
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        giou = iou - (enclose_area - union_area) / enclose_area
        return giou

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
            if self.giou:
                giou = self.box_giou(pred_boxes, y_true[..., :4]).type_as(x)
                loss_loc = torch.mean((1 - giou)[obj_mask])
            else:
                loss_x = torch.mean(self.BCELoss(x[obj_mask], y_true[..., 0][obj_mask]) * box_loss_scale[obj_mask])
                loss_y = torch.mean(self.BCELoss(y[obj_mask], y_true[..., 1][obj_mask]) * box_loss_scale[obj_mask])
                loss_w = torch.mean(self.MSELoss(w[obj_mask], y_true[..., 2][obj_mask]) * box_loss_scale[obj_mask])
                loss_h = torch.mean(self.MSELoss(h[obj_mask], y_true[..., 3][obj_mask]) * box_loss_scale[obj_mask])
                loss_loc = (loss_x + loss_y + loss_h + loss_w) * 0.1
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
            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_shapes), dim=-1)
            for t, best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[l]:
                    continue
                k = self.anchors_mask[l].index(best_n)
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()
                c = batch_target[t, 4].long()
                noobj_mask[b, k, j, i] = 0
                if not self.giou:
                    y_true[b, k, j, i, 0] = batch_target[t, 0] - i.float()
                    y_true[b, k, j, i, 1] = batch_target[t, 1] - j.float()
                    y_true[b, k, j, i, 2] = math.log(batch_target[t, 2] / anchors[best_n][0])
                    y_true[b, k, j, i, 3] = math.log(batch_target[t, 3] / anchors[best_n][1])
                    y_true[b, k, j, i, 4] = 1
                    y_true[b, k, j, i, c + 5] = 1
                else:
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
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DarkNet,
     lambda: ([], {'layers': [4, 4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_bubbliiiing_yolo3_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

