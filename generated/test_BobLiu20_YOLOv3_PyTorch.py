import sys
_module = sys.modules[__name__]
del sys
common = _module
coco_dataset = _module
data_transforms = _module
utils = _module
eval = _module
eval_coco = _module
params = _module
nets = _module
backbone = _module
darknet = _module
model_main = _module
yolo_loss = _module
test_fps = _module
test_images = _module
training = _module

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


import math


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import numpy as np


import logging


from collections import OrderedDict


import random


import torch.optim as optim


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1,
            padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=
            1, padding=1, bias=False)
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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1,
            padding=1, bias=False)
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
        layers.append(('ds_conv', nn.Conv2d(self.inplanes, planes[1],
            kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(('ds_bn', nn.BatchNorm2d(planes[1])))
        layers.append(('ds_relu', nn.LeakyReLU(0.1)))
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(('residual_{}'.format(i), BasicBlock(self.
                inplanes, planes)))
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


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, (0)] - box1[:, (2)] / 2, box1[:, (0)] + box1[:,
            (2)] / 2
        b1_y1, b1_y2 = box1[:, (1)] - box1[:, (3)] / 2, box1[:, (1)] + box1[:,
            (3)] / 2
        b2_x1, b2_x2 = box2[:, (0)] - box2[:, (2)] / 2, box2[:, (0)] + box2[:,
            (2)] / 2
        b2_y1, b2_y2 = box2[:, (1)] - box2[:, (3)] / 2, box2[:, (1)] + box2[:,
            (3)] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, (0)], box1[:, (1)], box1[:, (2)
            ], box1[:, (3)]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, (0)], box2[:, (1)], box2[:, (2)
            ], box2[:, (3)]
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0
        ) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


class YOLOLoss(nn.Module):

    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size
        self.ignore_threshold = 0.5
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, input, targets=None):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in
            self.anchors]
        prediction = input.view(bs, self.num_anchors, self.bbox_attrs, in_h,
            in_w).permute(0, 1, 3, 4, 2).contiguous()
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[(...), 5:])
        if targets is not None:
            mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.get_target(
                targets, scaled_anchors, in_w, in_h, self.ignore_threshold)
            mask, noobj_mask = mask, noobj_mask
            tx, ty, tw, th = tx, ty, tw, th
            tconf, tcls = tconf, tcls
            loss_x = self.bce_loss(x * mask, tx * mask)
            loss_y = self.bce_loss(y * mask, ty * mask)
            loss_w = self.mse_loss(w * mask, tw * mask)
            loss_h = self.mse_loss(h * mask, th * mask)
            loss_conf = self.bce_loss(conf * mask, mask) + 0.5 * self.bce_loss(
                conf * noobj_mask, noobj_mask * 0.0)
            loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1])
            loss = (loss_x * self.lambda_xy + loss_y * self.lambda_xy + 
                loss_w * self.lambda_wh + loss_h * self.lambda_wh + 
                loss_conf * self.lambda_conf + loss_cls * self.lambda_cls)
            return loss, loss_x.item(), loss_y.item(), loss_w.item(
                ), loss_h.item(), loss_conf.item(), loss_cls.item()
        else:
            FloatTensor = (torch.cuda.FloatTensor if x.is_cuda else torch.
                FloatTensor)
            LongTensor = (torch.cuda.LongTensor if x.is_cuda else torch.
                LongTensor)
            grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(
                bs * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t(
                ).repeat(bs * self.num_anchors, 1, 1).view(y.shape).type(
                FloatTensor)
            anchor_w = FloatTensor(scaled_anchors).index_select(1,
                LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1,
                LongTensor([1]))
            anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w
                .shape)
            anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h
                .shape)
            pred_boxes = FloatTensor(prediction[(...), :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
            _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
            output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale, conf.
                view(bs, -1, 1), pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data

    def get_target(self, target, anchors, in_w, in_h, ignore_threshold):
        bs = target.size(0)
        mask = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=
            False)
        noobj_mask = torch.ones(bs, self.num_anchors, in_h, in_w,
            requires_grad=False)
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad
            =False)
        tcls = torch.zeros(bs, self.num_anchors, in_h, in_w, self.
            num_classes, requires_grad=False)
        for b in range(bs):
            for t in range(target.shape[1]):
                if target[b, t].sum() == 0:
                    continue
                gx = target[b, t, 1] * in_w
                gy = target[b, t, 2] * in_h
                gw = target[b, t, 3] * in_w
                gh = target[b, t, 4] * in_h
                gi = int(gx)
                gj = int(gy)
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(
                    0)
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros(
                    (self.num_anchors, 2)), np.array(anchors)), 1))
                anch_ious = bbox_iou(gt_box, anchor_shapes)
                noobj_mask[b, anch_ious > ignore_threshold, gj, gi] = 0
                best_n = np.argmax(anch_ious)
                mask[b, best_n, gj, gi] = 1
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 
                    1e-16)
                th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 
                    1e-16)
                tconf[b, best_n, gj, gi] = 1
                tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1
        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_BobLiu20_YOLOv3_PyTorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(DarkNet(*[], **{'layers': [4, 4, 4, 4, 4]}), [torch.rand([4, 3, 64, 64])], {})

