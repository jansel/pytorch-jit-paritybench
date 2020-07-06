import sys
_module = sys.modules[__name__]
del sys
new_prune = _module
parse_config = _module
prune = _module
sparsity_train = _module
util = _module
yolomodel = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import torch.nn.functional as F


import random


import numpy as np


from torch.utils.data import Dataset


import torchvision.transforms as transforms


import torch.nn as nn


from torch.autograd import Variable


from torch.utils.data import DataLoader


from torchvision import datasets


from torchvision import transforms


from torch.optim import lr_scheduler


import math


from collections import defaultdict


class shortcutLayer(nn.Module):

    def __init__(self, froms):
        super(shortcutLayer, self).__init__()
        self.froms = froms


class Reorg(nn.Module):

    def __init__(self, stride):
        super(Reorg, self).__init__()
        self.stride = stride


class Route(nn.Module):

    def __init__(self, layers):
        super(Route, self).__init__()
        self.layers = layers


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
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


def build_targets(pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nG = grid_size
    mask = torch.zeros(nB, nA, nG, nG)
    conf_mask = torch.ones(nB, nA, nG, nG)
    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)
    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT += 1
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * nG
            gh = target[b, t, 4] * nG
            gi = int(gx)
            gj = int(gy)
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            best_n = np.argmax(anch_ious)
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                nCorrect += 1
    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls


class DetectionLayer(nn.Module):

    def __init__(self, anchors, num_classes, img_dim, ignore_thresh):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim
        self.ignore_thres = ignore_thresh
        self.lambda_coord = 1
        self.mse_loss = nn.MSELoss(reduction='elementwise_mean')
        self.bce_loss = nn.BCELoss(reduction='elementwise_mean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        nA = self.num_anchors
        nB = x.size(0)
        nG = x.size(2)
        stride = self.image_dim / nG
        FloatTensor = torch.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.ByteTensor if x.is_cuda else torch.ByteTensor
        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[(...), 5:])
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))
        pred_boxes = FloatTensor(prediction[(...), :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        if targets is not None:
            if x.is_cuda:
                self.mse_loss = self.mse_loss
                self.bce_loss = self.bce_loss
                self.ce_loss = self.ce_loss
            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(pred_boxes=pred_boxes.cpu().data, pred_conf=pred_conf.cpu().data, pred_cls=pred_cls.cpu().data, target=targets.cpu().data, anchors=scaled_anchors.cpu().data, num_anchors=nA, num_classes=self.num_classes, grid_size=nG, ignore_thres=self.ignore_thres, img_dim=self.image_dim)
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
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
            loss_cls = 1 / nB * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            return loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), recall, precision
        else:
            output = torch.cat((pred_boxes.view(nB, -1, 4) * stride, pred_conf.view(nB, -1, 1), pred_cls.view(nB, -1, self.num_classes)), -1)
            return output


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if x['type'] == 'convolutional':
            activation = x['activation']
            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            if batch_normalize:
                conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
                module.add_module('conv_with_bn_{0}'.format(index), conv)
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index), bn)
            else:
                conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
                module.add_module('conv_without_bn_{0}'.format(index), conv)
            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(index), activn)
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            module.add_module('upsample_{}'.format(index), upsample)
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = Route([start, end])
            module.add_module('route_{0}'.format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        elif x['type'] == 'shortcut':
            froms = int(x['from'])
            shortcut = shortcutLayer(froms)
            module.add_module('shortcut_{}'.format(index), shortcut)
        elif x['type'] == 'yolo' or x['type'] == 'region':
            try:
                mask = x['mask'].split(',')
                mask = [int(x) for x in mask]
                anchors = x['anchors'].split(',')
                anchors = [int(a) for a in anchors]
            except:
                mask = [int(x) for x in range(int(x['num']))]
                anchors = x['anchors'].split(',')
                anchors = [(32 * float(a)) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            num_classes = int(x['classes'])
            img_height = int(net_info['height'])
            try:
                ignore_thresh = float(x['ignore_thresh'])
            except:
                ignore_thresh = float(x['thresh'])
            detection = DetectionLayer(anchors, num_classes, img_height, ignore_thresh)
            module.add_module('Detection_{}'.format(index), detection)
        elif x['type'] == 'maxpool':
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            pool = nn.MaxPool2d(stride=stride, kernel_size=kernel_size)
            module.add_module('maxpool_{0}'.format(index), pool)
        elif x['type'] == 'reorg':
            stride = int(x['stride'])
            reorg = Reorg(stride=stride)
            module.add_module('reorg_{0}'.format(index), reorg)
            filters = filters * 4
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return net_info, module_list


def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]
    block = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    return blocks


class Darknet(nn.Module):

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.img_size = self.net_info['height']
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ['x', 'y', 'w', 'h', 'conf', 'cls', 'recall', 'precision']

    def forward(self, x, targets=None):
        is_training = targets is not None
        modules = self.blocks[1:]
        outputs = []
        layer_outputs = []
        self.losses = defaultdict(float)
        for i, module in enumerate(modules):
            module_type = module['type']
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)
            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i + from_] + outputs[i - 1]
            elif module_type == 'maxpool':
                x = self.module_list[i](x)
            elif module_type == 'reorg':
                stride = int(module['stride'])
                B, C, H, W = x.size()
                x = x.view(B, C, int(H / stride), stride, int(W / stride), stride).transpose(3, 4).contiguous()
                x = x.view(B, C, int(H / stride * W / stride), int(stride * stride)).transpose(2, 3).contiguous()
                x = x.view(B, C, int(stride * stride), int(H / stride), int(W / stride)).transpose(1, 2).contiguous()
                x = x.view(B, int(stride * stride * C), int(H / stride), int(W / stride))
            elif module_type == 'yolo' or module_type == 'region':
                if is_training:
                    x, *losses = self.module_list[i][0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                else:
                    x = self.module_list[i](x)
                layer_outputs.append(x)
            outputs.append(x)
        self.losses['recall'] /= 3
        self.losses['precision'] /= 3
        return sum(layer_outputs) if is_training else torch.cat(layer_outputs, 1)

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header_info = header
        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        fp.close()
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]['type']
            if module_type == 'convolutional':
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]['batch_normalize'])
                except:
                    batch_normalize = 0
                conv = model[0]
                if batch_normalize:
                    bn = model[1]
                    num_bn_biases = bn.bias.numel()
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    num_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
                    ptr = ptr + num_biases
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
        None

    def save_weights(self, path, cutoff=-1):
        """save layers between 0 and cutoff (cutoff = -1 -> all are saved)"""
        fp = open(path, 'wb')
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)
        for i in range(len(self.module_list[:cutoff])):
            module_type = self.blocks[i + 1]['type']
            if module_type == 'convolutional':
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]['batch_normalize'])
                except:
                    batch_normalize = 0
                conv = model[0]
                if batch_normalize:
                    bn = model[1]
                    bn.bias.data.cpu().numpy().tofile(fp)
                    bn.weight.data.cpu().numpy().tofile(fp)
                    bn.running_mean.data.cpu().numpy().tofile(fp)
                    bn.running_var.data.cpu().numpy().tofile(fp)
                else:
                    conv.bias.data.cpu().numpy().tofile(fp)
                conv.weight.data.cpu().numpy().tofile(fp)
        fp.close()

    def model_init(self):
        """init"""
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]['type']
            if module_type == 'convolutional':
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]['batch_normalize'])
                except:
                    batch_normalize = 0
                conv = model[0]
                if batch_normalize:
                    bn = model[1]
                    torch.nn.init.constant_(bn.weight.data, 0.5)
                    torch.nn.init.constant_(bn.bias.data, 0.0)
                else:
                    torch.nn.init.constant_(conv.bias.data, 0.0)
                torch.nn.init.xavier_uniform_(conv.weight.data, gain=1)

