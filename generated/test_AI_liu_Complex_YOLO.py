import sys
_module = sys.modules[__name__]
del sys
complexYOLO = _module
eval = _module
kitti = _module
main = _module
make_train_txt = _module
region_loss = _module
utils = _module

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


import torch.nn as nn


import torch.nn.functional as F


import torch


from torch.autograd import Variable


import numpy as np


import time


from scipy import misc


import torch.optim as optim


import torch.utils.data as data


import math


def reorg(x):
    stride = 2
    assert x.data.dim() == 4
    B = x.data.size(0)
    C = x.data.size(1)
    H = x.data.size(2)
    W = x.data.size(3)
    assert H % stride == 0
    assert W % stride == 0
    ws = stride
    hs = stride
    x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4
        ).contiguous()
    x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous(
        )
    x = x.view(B, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2
        ).contiguous()
    x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))
    return x


class ComplexYOLO(nn.Module):

    def __init__(self):
        super(ComplexYOLO, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size
            =3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(num_features=24)
        self.pool_1 = nn.MaxPool2d(2)
        self.conv_2 = nn.Conv2d(in_channels=24, out_channels=48,
            kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(num_features=48)
        self.pool_2 = nn.MaxPool2d(2)
        self.conv_3 = nn.Conv2d(in_channels=48, out_channels=64,
            kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm2d(num_features=64)
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=32,
            kernel_size=1, stride=1, padding=0)
        self.bn_4 = nn.BatchNorm2d(num_features=32)
        self.conv_5 = nn.Conv2d(in_channels=32, out_channels=64,
            kernel_size=3, stride=1, padding=1)
        self.bn_5 = nn.BatchNorm2d(num_features=64)
        self.pool_3 = nn.MaxPool2d(2)
        self.conv_6 = nn.Conv2d(in_channels=64, out_channels=128,
            kernel_size=3, stride=1, padding=1)
        self.bn_6 = nn.BatchNorm2d(num_features=128)
        self.conv_7 = nn.Conv2d(in_channels=128, out_channels=64,
            kernel_size=3, stride=1, padding=1)
        self.bn_7 = nn.BatchNorm2d(num_features=64)
        self.conv_8 = nn.Conv2d(in_channels=64, out_channels=128,
            kernel_size=3, stride=1, padding=1)
        self.bn_8 = nn.BatchNorm2d(num_features=128)
        self.pool_4 = nn.MaxPool2d(2)
        self.conv_9 = nn.Conv2d(in_channels=128, out_channels=256,
            kernel_size=3, stride=1, padding=1)
        self.bn_9 = nn.BatchNorm2d(num_features=256)
        self.conv_10 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=1, stride=1, padding=0)
        self.bn_10 = nn.BatchNorm2d(num_features=256)
        self.conv_11 = nn.Conv2d(in_channels=256, out_channels=512,
            kernel_size=3, stride=1, padding=1)
        self.bn_11 = nn.BatchNorm2d(num_features=512)
        self.pool_5 = nn.MaxPool2d(2)
        self.conv_12 = nn.Conv2d(in_channels=512, out_channels=512,
            kernel_size=3, stride=1, padding=1)
        self.bn_12 = nn.BatchNorm2d(num_features=512)
        self.conv_13 = nn.Conv2d(in_channels=512, out_channels=512,
            kernel_size=1, stride=1, padding=0)
        self.bn_13 = nn.BatchNorm2d(num_features=512)
        self.conv_14 = nn.Conv2d(in_channels=512, out_channels=1024,
            kernel_size=3, stride=1, padding=1)
        self.bn_14 = nn.BatchNorm2d(num_features=1024)
        self.conv_15 = nn.Conv2d(in_channels=1024, out_channels=1024,
            kernel_size=3, stride=1, padding=1)
        self.bn_15 = nn.BatchNorm2d(num_features=1024)
        self.conv_16 = nn.Conv2d(in_channels=1024, out_channels=1024,
            kernel_size=3, stride=1, padding=1)
        self.bn_16 = nn.BatchNorm2d(num_features=1024)
        self.conv_17 = nn.Conv2d(in_channels=2048, out_channels=1024,
            kernel_size=3, stride=1, padding=1)
        self.bn_17 = nn.BatchNorm2d(num_features=1024)
        self.conv_18 = nn.Conv2d(in_channels=1024, out_channels=75,
            kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.pool_1(x)
        x = self.relu(self.bn_2(self.conv_2(x)))
        x = self.pool_2(x)
        x = self.relu(self.bn_3(self.conv_3(x)))
        x = self.relu(self.bn_4(self.conv_4(x)))
        x = self.relu(self.bn_5(self.conv_5(x)))
        x = self.pool_3(x)
        x = self.relu(self.bn_6(self.conv_6(x)))
        x = self.relu(self.bn_7(self.conv_7(x)))
        x = self.relu(self.bn_8(self.conv_8(x)))
        x = self.pool_4(x)
        x = self.relu(self.bn_9(self.conv_9(x)))
        route_1 = x
        reorg_result = reorg(route_1)
        x = self.relu(self.bn_10(self.conv_10(x)))
        x = self.relu(self.bn_11(self.conv_11(x)))
        x = self.pool_5(x)
        x = self.relu(self.bn_12(self.conv_12(x)))
        x = self.relu(self.bn_13(self.conv_13(x)))
        x = self.relu(self.bn_14(self.conv_14(x)))
        x = self.relu(self.bn_15(self.conv_15(x)))
        x = self.relu(self.bn_16(self.conv_16(x)))
        x = torch.cat((reorg_result, x), 1)
        x = self.relu(self.bn_17(self.conv_17(x)))
        x = self.conv_18(x)
        return x


anchors = [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62,
    10.52]]


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


def build_targets(pred_boxes, pred_conf, pred_cls, target, anchors,
    num_anchors, num_classes, nH, nW, ignore_thres):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    mask = torch.zeros(nB, nA, nH, nW)
    conf_mask = torch.ones(nB, nA, nH, nW)
    tx = torch.zeros(nB, nA, nH, nW)
    ty = torch.zeros(nB, nA, nH, nW)
    tw = torch.zeros(nB, nA, nH, nW)
    tl = torch.zeros(nB, nA, nH, nW)
    tim = torch.zeros(nB, nA, nH, nW)
    tre = torch.zeros(nB, nA, nH, nW)
    tconf = torch.ByteTensor(nB, nA, nH, nW).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nH, nW, nC).fill_(0)
    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b][t].sum() == 0:
                continue
            nGT += 1
            gx = target[b, t, 1] * nW
            gy = target[b, t, 2] * nH
            gw = target[b, t, 3] * nW
            gl = target[b, t, 4] * nH
            gi = int(gx)
            gj = int(gy)
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gl])).unsqueeze(0)
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len
                (anchors), 2)), np.array(anchors)), 1))
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            best_n = np.argmax(anch_ious)
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gl])).unsqueeze(0)
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            tl[b, best_n, gj, gi] = math.log(gl / anchors[best_n][1] + 1e-16)
            target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                nCorrect += 1
    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, tl, tconf, tcls


class RegionLoss(nn.Module):

    def __init__(self, num_classes=8, num_anchors=5):
        super(RegionLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.bbox_attrs = 7 + num_classes
        self.ignore_thres = 0.6
        self.lambda_coord = 1
        self.mse_loss = nn.MSELoss(size_average=True)
        self.bce_loss = nn.BCELoss(size_average=True)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x, targets):
        nA = self.num_anchors
        nB = x.data.size(0)
        nH = x.data.size(2)
        nW = x.data.size(3)
        FloatTensor = (torch.cuda.FloatTensor if x.is_cuda else torch.
            FloatTensor)
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor
        prediction = x.view(nB, nA, self.bbox_attrs, nH, nW).permute(0, 1, 
            3, 4, 2).contiguous()
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 6])
        pred_cls = torch.sigmoid(prediction[(...), 7:])
        grid_x = torch.arange(nW).repeat(nH, 1).view([1, 1, nH, nW]).type(
            FloatTensor)
        grid_y = torch.arange(nH).repeat(nW, 1).t().view([1, 1, nH, nW]).type(
            FloatTensor)
        scaled_anchors = FloatTensor([(a_w, a_h) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))
        pred_boxes = FloatTensor(prediction[(...), :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        if x.is_cuda:
            self.mse_loss = self.mse_loss
            self.bce_loss = self.bce_loss
            self.ce_loss = self.ce_loss
        nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = (
            build_targets(pred_boxes=pred_boxes.cpu().data, pred_conf=
            pred_conf.cpu().data, pred_cls=pred_cls.cpu().data, target=
            targets.cpu().data, anchors=scaled_anchors.cpu().data,
            num_anchors=nA, num_classes=self.num_classes, nH=nH, nW=nW,
            ignore_thres=self.ignore_thres))
        nProposals = int((pred_conf > 0.5).sum().item())
        recall = float(nCorrect / nGT) if nGT else 1
        precision = float(nCorrect / nProposals)
        mask = Variable(mask.type(ByteTensor))
        conf_mask = Variable(conf_mask.type(ByteTensor))
        tx = Variable(tx.type(FloatTensor), requires_grad=False)
        ty = Variable(ty.type(FloatTensor), requires_grad=False)
        tw = Variable(tw.type(FloatTensor), requires_grad=False)
        th = Variable(th.type(FloatTensor), requires_grad=False)
        tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
        tcls = Variable(tcls.type(LongTensor), requires_grad=False)
        conf_mask_true = mask
        conf_mask_false = conf_mask - mask
        loss_x = self.mse_loss(x[mask], tx[mask])
        loss_y = self.mse_loss(y[mask], ty[mask])
        loss_w = self.mse_loss(w[mask], tw[mask])
        loss_h = self.mse_loss(h[mask], th[mask])
        loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[
            conf_mask_false]) + self.bce_loss(pred_conf[conf_mask_true],
            tconf[conf_mask_true])
        loss_cls = 1 / nB * self.ce_loss(pred_cls[mask], torch.argmax(tcls[
            mask], 1))
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        None
        return loss


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_AI_liu_Complex_YOLO(_paritybench_base):
    pass
    def test_000(self):
        self._check(ComplexYOLO(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

