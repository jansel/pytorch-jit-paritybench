import sys
_module = sys.modules[__name__]
del sys
voc2yolo3 = _module
get_dr_txt = _module
get_gt_txt = _module
get_map = _module
darknet = _module
yolo3 = _module
yolo_training = _module
predict = _module
test = _module
train = _module
config = _module
dataloader = _module
utils = _module
video = _module
voc_annotation = _module
yolo = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.backends.cudnn as cudnn


from torch.autograd import Variable


import math


from collections import OrderedDict


from random import shuffle


import torch.nn.functional as F


import time


import torch.optim as optim


from torch.utils.data import DataLoader


from torch.utils.data.dataset import Dataset


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


def darknet53(pretrained, **kwargs):
    model = DarkNet([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception('darknet request a pretrained path. got [{}]'.format(pretrained))
    return model


def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.ModuleList([conv2d(in_filters, filters_list[0], 1), conv2d(filters_list[0], filters_list[1], 3), conv2d(filters_list[1], filters_list[0], 1), conv2d(filters_list[0], filters_list[1], 3), conv2d(filters_list[1], filters_list[0], 1), conv2d(filters_list[0], filters_list[1], 3), nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)])
    return m


class YoloBody(nn.Module):

    def __init__(self, config):
        super(YoloBody, self).__init__()
        self.config = config
        self.backbone = darknet53(None)
        out_filters = self.backbone.layers_out_filters
        final_out_filter0 = len(config['yolo']['anchors'][0]) * (5 + config['yolo']['classes'])
        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], final_out_filter0)
        final_out_filter1 = len(config['yolo']['anchors'][1]) * (5 + config['yolo']['classes'])
        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, final_out_filter1)
        final_out_filter2 = len(config['yolo']['anchors'][2]) * (5 + config['yolo']['classes'])
        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, final_out_filter2)

    def forward(self, x):

        def _branch(last_layer, layer_in):
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
                if i == 4:
                    out_branch = layer_in
            return layer_in, out_branch
        x2, x1, x0 = self.backbone(x)
        out0, out0_branch = _branch(self.last_layer0, x0)
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.last_layer1, x1_in)
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, _ = _branch(self.last_layer2, x2_in)
        return out0, out1, out2


def clip_by_tensor(t, t_min, t_max):
    t = t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def BCELoss(pred, target):
    epsilon = 1e-07
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


def MSELoss(pred, target):
    return (pred - target) ** 2


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
        计算IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, (0)] - box1[:, (2)] / 2, box1[:, (0)] + box1[:, (2)] / 2
        b1_y1, b1_y2 = box1[:, (1)] - box1[:, (3)] / 2, box1[:, (1)] + box1[:, (3)] / 2
        b2_x1, b2_x2 = box2[:, (0)] - box2[:, (2)] / 2, box2[:, (0)] + box2[:, (2)] / 2
        b2_y1, b2_y2 = box2[:, (1)] - box2[:, (3)] / 2, box2[:, (1)] + box2[:, (3)] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, (0)], box1[:, (1)], box1[:, (2)], box1[:, (3)]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, (0)], box2[:, (1)], box2[:, (2)], box2[:, (3)]
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def jaccard(_box_a, _box_b):
    b1_x1, b1_x2 = _box_a[:, (0)] - _box_a[:, (2)] / 2, _box_a[:, (0)] + _box_a[:, (2)] / 2
    b1_y1, b1_y2 = _box_a[:, (1)] - _box_a[:, (3)] / 2, _box_a[:, (1)] + _box_a[:, (3)] / 2
    b2_x1, b2_x2 = _box_b[:, (0)] - _box_b[:, (2)] / 2, _box_b[:, (0)] + _box_b[:, (2)] / 2
    b2_y1, b2_y2 = _box_b[:, (1)] - _box_b[:, (3)] / 2, _box_b[:, (1)] + _box_b[:, (3)] / 2
    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)
    box_a[:, (0)], box_a[:, (1)], box_a[:, (2)], box_a[:, (3)] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, (0)], box_b[:, (1)], box_b[:, (2)], box_b[:, (3)] = b2_x1, b2_y1, b2_x2, b2_y2
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)
    inter = inter[:, :, (0)] * inter[:, :, (1)]
    area_a = ((box_a[:, (2)] - box_a[:, (0)]) * (box_a[:, (3)] - box_a[:, (1)])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, (2)] - box_b[:, (0)]) * (box_b[:, (3)] - box_b[:, (1)])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


class YOLOLoss(nn.Module):

    def __init__(self, anchors, num_classes, img_size, cuda):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.feature_length = [img_size[0] // 32, img_size[0] // 16, img_size[0] // 8]
        self.img_size = img_size
        self.ignore_threshold = 0.5
        self.lambda_xy = 1.0
        self.lambda_wh = 1.0
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.cuda = cuda

    def forward(self, input, targets=None):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        prediction = input.view(bs, int(self.num_anchors / 3), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[(...), 5:])
        mask, noobj_mask, tx, ty, tw, th, tconf, tcls, box_loss_scale_x, box_loss_scale_y = self.get_target(targets, scaled_anchors, in_w, in_h, self.ignore_threshold)
        noobj_mask = self.get_ignore(prediction, targets, scaled_anchors, in_w, in_h, noobj_mask)
        if self.cuda:
            box_loss_scale_x = box_loss_scale_x
            box_loss_scale_y = box_loss_scale_y
            mask, noobj_mask = mask, noobj_mask
            tx, ty, tw, th = tx, ty, tw, th
            tconf, tcls = tconf, tcls
        box_loss_scale = 2 - box_loss_scale_x * box_loss_scale_y
        loss_x = torch.sum(BCELoss(x, tx) / bs * box_loss_scale * mask)
        loss_y = torch.sum(BCELoss(y, ty) / bs * box_loss_scale * mask)
        loss_w = torch.sum(MSELoss(w, tw) / bs * 0.5 * box_loss_scale * mask)
        loss_h = torch.sum(MSELoss(h, th) / bs * 0.5 * box_loss_scale * mask)
        loss_conf = torch.sum(BCELoss(conf, mask) * mask / bs) + torch.sum(BCELoss(conf, mask) * noobj_mask / bs)
        loss_cls = torch.sum(BCELoss(pred_cls[mask == 1], tcls[mask == 1]) / bs)
        loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + loss_w * self.lambda_wh + loss_h * self.lambda_wh + loss_conf * self.lambda_conf + loss_cls * self.lambda_cls
        return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item()

    def get_target(self, target, anchors, in_w, in_h, ignore_threshold):
        bs = len(target)
        anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]][self.feature_length.index(in_w)]
        subtract_index = [0, 3, 6][self.feature_length.index(in_w)]
        mask = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        tx = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, self.num_classes, requires_grad=False)
        box_loss_scale_x = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        box_loss_scale_y = torch.zeros(bs, int(self.num_anchors / 3), in_h, in_w, requires_grad=False)
        for b in range(bs):
            for t in range(target[b].shape[0]):
                gx = target[b][t, 0] * in_w
                gy = target[b][t, 1] * in_h
                gw = target[b][t, 2] * in_w
                gh = target[b][t, 3] * in_h
                gi = int(gx)
                gj = int(gy)
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)), np.array(anchors)), 1))
                anch_ious = bbox_iou(gt_box, anchor_shapes)
                best_n = np.argmax(anch_ious)
                if best_n not in anchor_index:
                    continue
                if gj < in_h and gi < in_w:
                    best_n = best_n - subtract_index
                    noobj_mask[b, best_n, gj, gi] = 0
                    mask[b, best_n, gj, gi] = 1
                    tx[b, best_n, gj, gi] = gx - gi
                    ty[b, best_n, gj, gi] = gy - gj
                    tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n + subtract_index][0])
                    th[b, best_n, gj, gi] = math.log(gh / anchors[best_n + subtract_index][1])
                    box_loss_scale_x[b, best_n, gj, gi] = target[b][t, 2]
                    box_loss_scale_y[b, best_n, gj, gi] = target[b][t, 3]
                    tconf[b, best_n, gj, gi] = 1
                    tcls[b, best_n, gj, gi, int(target[b][t, 4])] = 1
                else:
                    None
                    None
                    continue
        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls, box_loss_scale_x, box_loss_scale_y

    def get_ignore(self, prediction, target, scaled_anchors, in_w, in_h, noobj_mask):
        bs = len(target)
        anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]][self.feature_length.index(in_w)]
        scaled_anchors = np.array(scaled_anchors)[anchor_index]
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        FloatTensor = torch.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.LongTensor if x.is_cuda else torch.LongTensor
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(int(bs * self.num_anchors / 3), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(int(bs * self.num_anchors / 3), 1, 1).view(y.shape).type(FloatTensor)
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        pred_boxes = FloatTensor(prediction[(...), :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        for i in range(bs):
            pred_boxes_for_ignore = pred_boxes[i]
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)
            if len(target[i]) > 0:
                gx = target[i][:, 0:1] * in_w
                gy = target[i][:, 1:2] * in_h
                gw = target[i][:, 2:3] * in_w
                gh = target[i][:, 3:4] * in_h
                gt_box = torch.FloatTensor(np.concatenate([gx, gy, gw, gh], -1)).type(FloatTensor)
                anch_ious = jaccard(gt_box, pred_boxes_for_ignore)
                for t in range(target[i].shape[0]):
                    anch_iou = anch_ious[t].view(pred_boxes[i].size()[:3])
                    noobj_mask[i][anch_iou > self.ignore_threshold] = 0
        return noobj_mask


class DecodeBox(nn.Module):

    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self, input):
        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors]
        prediction = input.view(batch_size, self.num_anchors, self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[(...), 5:])
        FloatTensor = torch.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.LongTensor if x.is_cuda else torch.LongTensor
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_width, 1).repeat(batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_height, 1).t().repeat(batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)
        pred_boxes = FloatTensor(prediction[(...), :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale, conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output.data


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

