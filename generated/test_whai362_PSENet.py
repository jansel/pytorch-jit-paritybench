import sys
_module = sys.modules[__name__]
del sys
python3 = _module
psenet_r50_ctw = _module
psenet_r50_ctw_finetune = _module
psenet_r50_ic15_1024 = _module
psenet_r50_ic15_1024_finetune = _module
psenet_r50_ic15_736 = _module
psenet_r50_ic15_736_finetune = _module
psenet_r50_synth = _module
psenet_r50_tt = _module
psenet_r50_tt_finetune = _module
dataset = _module
builder = _module
psenet = _module
check_dataloader = _module
psenet_ctw = _module
psenet_ic15 = _module
psenet_synth = _module
psenet_tt = _module
eval = _module
file_util = _module
rrc_evaluation_funcs = _module
rrc_evaluation_funcs_v1 = _module
rrc_evaluation_funcs_v2 = _module
script = _module
script_self_adapt = _module
rrc_evaluation_funcs_1_1 = _module
Deteval = _module
Deteval_rec = _module
polygon_wrapper = _module
models = _module
backbone = _module
resnet = _module
head = _module
psenet_head = _module
loss = _module
acc = _module
dice_loss = _module
emb_loss_v1 = _module
iou = _module
ohem = _module
neck = _module
fpn = _module
post_processing = _module
pse = _module
setup = _module
psenet = _module
pypse = _module
utils = _module
conv_bn_relu = _module
fuse_conv_bn = _module
test = _module
train = _module
average_meter = _module
logger = _module
result_format = _module

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


import numpy as np


import random


from torch.utils import data


import torchvision.transforms as transforms


import math


import string


import scipy.io as scio


import torch.nn as nn


import torch.nn.functional as F


import time


from torch.autograd import Function


from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
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


class Convkxk(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Convkxk, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResNet(nn.Module):

    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 128
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
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
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        f = []
        x = self.layer1(x)
        f.append(x)
        x = self.layer2(x)
        f.append(x)
        x = self.layer3(x)
        f.append(x)
        x = self.layer4(x)
        f.append(x)
        return tuple(f)


def build_loss(cfg):
    param = dict()
    for key in cfg:
        if key == 'type':
            continue
        param[key] = cfg[key]
    loss = models.loss.__dict__[cfg.type](**param)
    return loss


EPS = 1e-06


def iou_single(a, b, mask, n_class):
    valid = mask == 1
    a = a[valid]
    b = b[valid]
    miou = []
    for i in range(n_class):
        inter = ((a == i) & (b == i)).float()
        union = ((a == i) | (b == i)).float()
        miou.append(torch.sum(inter) / (torch.sum(union) + EPS))
    miou = sum(miou) / len(miou)
    return miou


def iou(a, b, mask, n_class=2, reduce=True):
    batch_size = a.size(0)
    a = a.view(batch_size, -1)
    b = b.view(batch_size, -1)
    mask = mask.view(batch_size, -1)
    iou = a.new_zeros((batch_size,), dtype=torch.float32)
    for i in range(batch_size):
        iou[i] = iou_single(a[i], b[i], mask[i], n_class)
    if reduce:
        iou = torch.mean(iou)
    return iou


def ohem_single(score, gt_text, training_mask):
    pos_num = int(torch.sum(gt_text > 0.5)) - int(torch.sum((gt_text > 0.5) & (training_mask <= 0.5)))
    if pos_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.view(1, selected_mask.shape[0], selected_mask.shape[1]).float()
        return selected_mask
    neg_num = int(torch.sum(gt_text <= 0.5))
    neg_num = int(min(pos_num * 3, neg_num))
    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.view(1, selected_mask.shape[0], selected_mask.shape[1]).float()
        return selected_mask
    neg_score = score[gt_text <= 0.5]
    neg_score_sorted, _ = torch.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]
    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).float()
    return selected_mask


def ohem_batch(scores, gt_texts, training_masks):
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))
    selected_masks = torch.cat(selected_masks, 0).float()
    return selected_masks


def pse(kernals, min_area):
    kernal_num = len(kernals)
    pred = np.zeros(kernals[0].shape, dtype='int32')
    label_num, label = cv2.connectedComponents(kernals[kernal_num - 1], connectivity=4)
    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0
    queue = Queue(maxsize=0)
    next_queue = Queue(maxsize=0)
    points = np.array(np.where(label > 0)).transpose((1, 0))
    for point_idx in range(points.shape[0]):
        x, y = points[point_idx, 0], points[point_idx, 1]
        l = label[x, y]
        queue.put((x, y, l))
        pred[x, y] = l
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    for kernal_idx in range(kernal_num - 2, -1, -1):
        kernal = kernals[kernal_idx].copy()
        while not queue.empty():
            x, y, l = queue.get()
            is_edge = True
            for j in range(4):
                tmpx = x + dx[j]
                tmpy = y + dy[j]
                if tmpx < 0 or tmpx >= kernal.shape[0] or tmpy < 0 or tmpy >= kernal.shape[1]:
                    continue
                if kernal[tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:
                    continue
                queue.put((tmpx, tmpy, l))
                pred[tmpx, tmpy] = l
                is_edge = False
            if is_edge:
                next_queue.put((x, y, l))
        queue, next_queue = next_queue, queue
    return pred


class PSENet_Head(nn.Module):

    def __init__(self, in_channels, hidden_dim, num_classes, loss_text, loss_kernel):
        super(PSENet_Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, num_classes, kernel_size=1, stride=1, padding=0)
        self.text_loss = build_loss(loss_text)
        self.kernel_loss = build_loss(loss_kernel)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f):
        out = self.conv1(f)
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)
        return out

    def get_results(self, out, img_meta, cfg):
        outputs = dict()
        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()
        score = torch.sigmoid(out[:, 0, :, :])
        kernels = out[:, :cfg.test_cfg.kernel_num, :, :] > 0
        text_mask = kernels[:, :1, :, :]
        kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask
        score = score.data.cpu().numpy()[0].astype(np.float32)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
        label = pse(kernels, cfg.test_cfg.min_area)
        org_img_size = img_meta['org_img_size'][0]
        img_size = img_meta['img_size'][0]
        label_num = np.max(label) + 1
        label = cv2.resize(label, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        score = cv2.resize(score, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(det_pse_time=time.time() - start))
        scale = float(org_img_size[1]) / float(img_size[1]), float(org_img_size[0]) / float(img_size[0])
        bboxes = []
        scores = []
        for i in range(1, label_num):
            ind = label == i
            points = np.array(np.where(ind)).transpose((1, 0))
            if points.shape[0] < cfg.test_cfg.min_area:
                label[ind] = 0
                continue
            score_i = np.mean(score[ind])
            if score_i < cfg.test_cfg.min_score:
                label[ind] = 0
                continue
            if cfg.test_cfg.bbox_type == 'rect':
                rect = cv2.minAreaRect(points[:, ::-1])
                bbox = cv2.boxPoints(rect) * scale
            elif cfg.test_cfg.bbox_type == 'poly':
                binary = np.zeros(label.shape, dtype='uint8')
                binary[ind] = 1
                _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bbox = contours[0] * scale
            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))
            scores.append(score_i)
        outputs.update(dict(bboxes=bboxes, scores=scores))
        return outputs

    def loss(self, out, gt_texts, gt_kernels, training_masks):
        texts = out[:, 0, :, :]
        kernels = out[:, 1:, :, :]
        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        loss_text = self.text_loss(texts, gt_texts, selected_masks, reduce=False)
        iou_text = iou((texts > 0).long(), gt_texts, training_masks, reduce=False)
        losses = dict(loss_text=loss_text, iou_text=iou_text)
        loss_kernels = []
        selected_masks = gt_texts * training_masks
        for i in range(kernels.size(1)):
            kernel_i = kernels[:, i, :, :]
            gt_kernel_i = gt_kernels[:, i, :, :]
            loss_kernel_i = self.kernel_loss(kernel_i, gt_kernel_i, selected_masks, reduce=False)
            loss_kernels.append(loss_kernel_i)
        loss_kernels = torch.mean(torch.stack(loss_kernels, dim=1), dim=1)
        iou_kernel = iou((kernels[:, -1, :, :] > 0).long(), gt_kernels[:, -1, :, :], training_masks * gt_texts, reduce=False)
        losses.update(dict(loss_kernels=loss_kernels, iou_kernel=iou_kernel))
        return losses


class DiceLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input, target, mask, reduce=True):
        batch_size = input.size(0)
        input = torch.sigmoid(input)
        input = input.contiguous().view(batch_size, -1)
        target = target.contiguous().view(batch_size, -1).float()
        mask = mask.contiguous().view(batch_size, -1).float()
        input = input * mask
        target = target * mask
        a = torch.sum(input * target, dim=1)
        b = torch.sum(input * input, dim=1) + 0.001
        c = torch.sum(target * target, dim=1) + 0.001
        d = 2 * a / (b + c)
        loss = 1 - d
        loss = self.loss_weight * loss
        if reduce:
            loss = torch.mean(loss)
        return loss


class EmbLoss_v1(nn.Module):

    def __init__(self, feature_dim=4, loss_weight=1.0):
        super(EmbLoss_v1, self).__init__()
        self.feature_dim = feature_dim
        self.loss_weight = loss_weight
        self.delta_v = 0.5
        self.delta_d = 1.5
        self.weights = 1.0, 1.0

    def forward_single(self, emb, instance, kernel, training_mask, bboxes):
        training_mask = (training_mask > 0.5).long()
        kernel = (kernel > 0.5).long()
        instance = instance * training_mask
        instance_kernel = (instance * kernel).view(-1)
        instance = instance.view(-1)
        emb = emb.view(self.feature_dim, -1)
        unique_labels, unique_ids = torch.unique(instance_kernel, sorted=True, return_inverse=True)
        num_instance = unique_labels.size(0)
        if num_instance <= 1:
            return 0
        emb_mean = emb.new_zeros((self.feature_dim, num_instance), dtype=torch.float32)
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind_k = instance_kernel == lb
            emb_mean[:, i] = torch.mean(emb[:, ind_k], dim=1)
        l_agg = emb.new_zeros(num_instance, dtype=torch.float32)
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind = instance == lb
            emb_ = emb[:, ind]
            dist = (emb_ - emb_mean[:, i:i + 1]).norm(p=2, dim=0)
            dist = F.relu(dist - self.delta_v) ** 2
            l_agg[i] = torch.mean(torch.log(dist + 1.0))
        l_agg = torch.mean(l_agg[1:])
        if num_instance > 2:
            emb_interleave = emb_mean.permute(1, 0).repeat(num_instance, 1)
            emb_band = emb_mean.permute(1, 0).repeat(1, num_instance).view(-1, self.feature_dim)
            mask = (1 - torch.eye(num_instance, dtype=torch.int8)).view(-1, 1).repeat(1, self.feature_dim)
            mask = mask.view(num_instance, num_instance, -1)
            mask[0, :, :] = 0
            mask[:, 0, :] = 0
            mask = mask.view(num_instance * num_instance, -1)
            dist = emb_interleave - emb_band
            dist = dist[mask > 0].view(-1, self.feature_dim).norm(p=2, dim=1)
            dist = F.relu(2 * self.delta_d - dist) ** 2
            l_dis = torch.mean(torch.log(dist + 1.0))
        else:
            l_dis = 0
        l_agg = self.weights[0] * l_agg
        l_dis = self.weights[1] * l_dis
        l_reg = torch.mean(torch.log(torch.norm(emb_mean, 2, 0) + 1.0)) * 0.001
        loss = l_agg + l_dis + l_reg
        return loss

    def forward(self, emb, instance, kernel, training_mask, bboxes, reduce=True):
        loss_batch = emb.new_zeros(emb.size(0), dtype=torch.float32)
        for i in range(loss_batch.size(0)):
            loss_batch[i] = self.forward_single(emb[i], instance[i], kernel[i], training_mask[i], bboxes[i])
        loss_batch = self.loss_weight * loss_batch
        if reduce:
            loss_batch = torch.mean(loss_batch)
        return loss_batch


class Conv_BN_ReLU(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FPN(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        self.toplayer_ = Conv_BN_ReLU(2048, 256, kernel_size=1, stride=1, padding=0)
        self.smooth1_ = Conv_BN_ReLU(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_ = Conv_BN_ReLU(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_ = Conv_BN_ReLU(256, 256, kernel_size=3, stride=1, padding=1)
        self.latlayer1_ = Conv_BN_ReLU(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2_ = Conv_BN_ReLU(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3_ = Conv_BN_ReLU(256, 256, kernel_size=1, stride=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, f2, f3, f4, f5):
        p5 = self.toplayer_(f5)
        f4 = self.latlayer1_(f4)
        p4 = self._upsample_add(p5, f4)
        p4 = self.smooth1_(p4)
        f3 = self.latlayer2_(f3)
        p3 = self._upsample_add(p4, f3)
        p3 = self.smooth2_(p3)
        f2 = self.latlayer3_(f2)
        p2 = self._upsample_add(p3, f2)
        p2 = self.smooth3_(p2)
        p3 = self._upsample(p3, p2)
        p4 = self._upsample(p4, p2)
        p5 = self._upsample(p5, p2)
        return p2, p3, p4, p5


def build_backbone(cfg):
    param = dict()
    for key in cfg:
        if key == 'type':
            continue
        param[key] = cfg[key]
    backbone = models.backbone.__dict__[cfg.type](**param)
    return backbone


def build_head(cfg):
    param = dict()
    for key in cfg:
        if key == 'type':
            continue
        param[key] = cfg[key]
    head = models.head.__dict__[cfg.type](**param)
    return head


def build_neck(cfg):
    param = dict()
    for key in cfg:
        if key == 'type':
            continue
        param[key] = cfg[key]
    neck = models.neck.__dict__[cfg.type](**param)
    return neck


class PSENet(nn.Module):

    def __init__(self, backbone, neck, detection_head):
        super(PSENet, self).__init__()
        self.backbone = build_backbone(backbone)
        self.fpn = build_neck(neck)
        self.det_head = build_head(detection_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self, imgs, gt_texts=None, gt_kernels=None, training_masks=None, img_metas=None, cfg=None):
        outputs = dict()
        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()
        f = self.backbone(imgs)
        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(backbone_time=time.time() - start))
            start = time.time()
        f1, f2, f3, f4 = self.fpn(f[0], f[1], f[2], f[3])
        f = torch.cat((f1, f2, f3, f4), 1)
        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(neck_time=time.time() - start))
            start = time.time()
        det_out = self.det_head(f)
        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(det_head_time=time.time() - start))
        if self.training:
            det_out = self._upsample(det_out, imgs.size())
            det_loss = self.det_head.loss(det_out, gt_texts, gt_kernels, training_masks)
            outputs.update(det_loss)
        else:
            det_out = self._upsample(det_out, imgs.size(), 1)
            det_res = self.det_head.get_results(det_out, img_metas, cfg)
            outputs.update(det_res)
        return outputs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv_BN_ReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Convkxk,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (FPN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 256, 64, 64]), torch.rand([4, 512, 64, 64]), torch.rand([4, 1024, 64, 64]), torch.rand([4, 2048, 64, 64])], {}),
     False),
]

class Test_whai362_PSENet(_paritybench_base):
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

