import sys
_module = sys.modules[__name__]
del sys
default_path = _module
detect = _module
inference = _module
darknet = _module
yolo_nano_helper = _module
test = _module
train = _module
utils = _module
augmentations = _module
board = _module
dataloader_utils = _module
datasets = _module
logger = _module
mc_reader = _module
optim = _module
parse_config = _module
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


import time


import torch


from torch.utils.data import DataLoader


from torchvision import datasets


from torch.autograd import Variable


import matplotlib.pyplot as plt


import matplotlib.patches as patches


from matplotlib.ticker import NullLocator


from torch.nn.parallel import DataParallel


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import torch.utils.model_zoo as model_zoo


from torchvision import transforms


import torch.optim as optim


import random


from torch.utils.data import Dataset


import torchvision.transforms as transforms


from collections import defaultdict


import math


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = w1 * h1 + 1e-16 + w2 * h2 - inter_area
    return inter_area / union_area


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres, img_scores=None, gt_mix_index=None):
    ByteTensor = torch.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    sum_weights_1 = FloatTensor(nB, nA, nG, nG).fill_(0)
    sum_weights_2 = FloatTensor(nB, nA, nG, nG).fill_(0)
    img_scores_1 = img_scores[gt_mix_index == 0]
    val, arg_index = torch.unique(img_scores_1, return_inverse=True)
    arg_index = arg_index.cpu().numpy().tolist()
    sorted_index = list(set(arg_index))
    sorted_index.sort(key=arg_index.index)
    batches_weight_1 = val[sorted_index][:nB, None, None, None]
    batches_weight_1 = batches_weight_1.expand_as(sum_weights_1)
    img_scores_2 = img_scores[gt_mix_index == 1]
    if img_scores_2.shape[0] == 0:
        batches_weight_2 = None
    else:
        val, arg_index = torch.unique(img_scores_2, return_inverse=True)
        arg_index = arg_index.cpu().numpy().tolist()
        sorted_index = list(set(arg_index))
        sorted_index.sort(key=arg_index.index)
        batches_weight_2 = val[sorted_index][:nB, None, None, None]
        batches_weight_2 = batches_weight_2.expand_as(sum_weights_2)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    tcls[b, best_n, gj, gi, target_labels] = 1
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)
    tconf = obj_mask.float()
    gxy_1 = gxy[gt_mix_index == 0]
    gi_1, gj_1 = gxy_1.long().t()
    b_1 = b[gt_mix_index == 0]
    best_n_1 = best_n[gt_mix_index == 0]
    obj_mask_1 = ByteTensor(nB, nA, nG, nG).fill_(0)
    obj_mask_1[b_1, best_n_1, gj_1, gi_1] = 1
    gxy_2 = gxy[gt_mix_index == 1]
    if gxy_2.shape[0] == 0:
        obj_mask_2 = None
    else:
        gi_2, gj_2 = gxy_2.long().t()
        b_2 = b[gt_mix_index == 1]
        best_n_2 = best_n[gt_mix_index == 1]
        obj_mask_2 = ByteTensor(nB, nA, nG, nG).fill_(0)
        obj_mask_2[b_2, best_n_2, gj_2, gi_2] = 1
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf, [batches_weight_1, batches_weight_2], [obj_mask_1, obj_mask_2]


def to_cpu(tensor):
    return tensor.detach().cpu()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None, img_scores=None, gt_mix_index=None):
        FloatTensor = torch.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.ByteTensor if x.is_cuda else torch.ByteTensor
        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)
        prediction = x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
        output = torch.cat((pred_boxes.view(num_samples, -1, 4) * self.stride, pred_conf.view(num_samples, -1, 1), pred_cls.view(num_samples, -1, self.num_classes)), -1)
        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf, batches_weights, obj_mask_mix_index = build_targets(pred_boxes=pred_boxes, pred_cls=pred_cls, target=targets, anchors=self.scaled_anchors, ignore_thres=self.ignore_thres, img_scores=img_scores, gt_mix_index=gt_mix_index)
            sum_weights1, sum_weights2 = batches_weights
            obj_mask_1, obj_mask_2 = obj_mask_mix_index
            loss_x_1 = torch.mean(self.mse_loss(x[obj_mask_1], tx[obj_mask_1]) * sum_weights1[obj_mask_1])
            loss_y_1 = torch.mean(self.mse_loss(y[obj_mask_1], ty[obj_mask_1]) * sum_weights1[obj_mask_1])
            loss_w_1 = torch.mean(self.mse_loss(w[obj_mask_1], tw[obj_mask_1]) * sum_weights1[obj_mask_1])
            loss_h_1 = torch.mean(self.mse_loss(h[obj_mask_1], th[obj_mask_1]) * sum_weights1[obj_mask_1])
            if obj_mask_2 is not None:
                loss_x_2 = torch.mean(self.mse_loss(x[obj_mask_2], tx[obj_mask_2]) * sum_weights2[obj_mask_2])
                loss_y_2 = torch.mean(self.mse_loss(y[obj_mask_2], ty[obj_mask_2]) * sum_weights2[obj_mask_2])
                loss_w_2 = torch.mean(self.mse_loss(w[obj_mask_2], tw[obj_mask_2]) * sum_weights2[obj_mask_2])
                loss_h_2 = torch.mean(self.mse_loss(h[obj_mask_2], th[obj_mask_2]) * sum_weights2[obj_mask_2])
            else:
                loss_x_2 = 0.0
                loss_y_2 = 0.0
                loss_w_2 = 0.0
                loss_h_2 = 0.0
            loss_x = loss_x_1 + loss_x_2
            loss_y = loss_y_1 + loss_y_2
            loss_w = loss_w_1 + loss_w_2
            loss_h = loss_h_1 + loss_h_2
            loss_conf_obj_1 = torch.mean(self.bce_loss(pred_conf[obj_mask_1], tconf[obj_mask_1]) * sum_weights1[obj_mask_1])
            if obj_mask_2 is not None:
                loss_conf_obj_2 = torch.mean(self.bce_loss(pred_conf[obj_mask_2], tconf[obj_mask_2]) * sum_weights2[obj_mask_2])
            else:
                loss_conf_obj_2 = 0.0
            loss_conf_obj = loss_conf_obj_1 + loss_conf_obj_2
            loss_conf_noobj = torch.mean(self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask]))
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls_1 = torch.mean(self.bce_loss(pred_cls[obj_mask_1], tcls[obj_mask_1]) * sum_weights1[obj_mask_1])
            if obj_mask_2 is not None:
                loss_cls_2 = torch.mean(self.bce_loss(pred_cls[obj_mask_2], tcls[obj_mask_2]) * sum_weights2[obj_mask_2])
            else:
                loss_cls_2 = 0.0
            loss_cls = loss_cls_1 + loss_cls_2
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)
            self.metrics = {'loss': to_cpu(total_loss).item(), 'x': to_cpu(loss_x).item(), 'y': to_cpu(loss_y).item(), 'w': to_cpu(loss_w).item(), 'h': to_cpu(loss_h).item(), 'conf': to_cpu(loss_conf).item(), 'cls': to_cpu(loss_cls).item(), 'cls_acc': to_cpu(cls_acc).item(), 'recall50': to_cpu(recall50).item(), 'recall75': to_cpu(recall75).item(), 'precision': to_cpu(precision).item(), 'conf_obj': to_cpu(conf_obj).item(), 'conf_noobj': to_cpu(conf_noobj).item(), 'grid_size': grid_size}
            return output, total_loss


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2
            modules.add_module(f'conv_{module_i}', nn.Conv2d(in_channels=output_filters[-1], out_channels=filters, kernel_size=kernel_size, stride=int(module_def['stride']), padding=pad, bias=not bn))
            if bn:
                modules.add_module(f'batch_norm_{module_i}', nn.BatchNorm2d(filters, momentum=0.9, eps=1e-05))
            if module_def['activation'] == 'leaky':
                modules.add_module(f'leaky_{module_i}', nn.LeakyReLU(0.1))
        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module(f'_debug_padding_{module_i}', nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module(f'maxpool_{module_i}', maxpool)
        elif module_def['type'] == 'upsample':
            upsample = Upsample(scale_factor=int(module_def['stride']), mode='nearest')
            modules.add_module(f'upsample_{module_i}', upsample)
        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f'route_{module_i}', EmptyLayer())
        elif module_def['type'] == 'shortcut':
            filters = output_filters[1:][int(module_def['from'])]
            modules.add_module(f'shortcut_{module_i}', EmptyLayer())
        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            anchors = [int(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def['classes'])
            img_size = int(hyperparams['height'])
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f'yolo_{module_i}', yolo_layer)
        module_list.append(modules)
        output_filters.append(filters)
    return hyperparams, module_list


def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    module_defs = []
    for line in lines:
        if line.startswith('['):
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split('=')
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
    return module_defs


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], 'metrics')]
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif module_def['type'] == 'route':
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def['layers'].split(',')], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""
        with open(weights_path, 'rb') as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header_info = header
            self.seen = header[3]
            weights = np.fromfile(f, dtype=np.float32)
        cutoff = None
        if 'darknet53.conv.74' in weights_path:
            cutoff = 75
        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_darknet_weights(self, path, cutoff=-1):
        """
            @:param path    - path of the new weights file
            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, 'wb')
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                conv_layer.weight.data.cpu().numpy().tofile(fp)
        fp.close()


ACTIVATE = {'relu': nn.ReLU, 'relu6': nn.ReLU6, 'leaky': nn.LeakyReLU}


class conv3x3(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, groups=1, dilation=1, act='leaky'):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = ACTIVATE[act](inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class conv1x1(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, groups=1, dilation=1, act='leaky', use_relu=True):
        super(conv1x1, self).__init__()
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if use_relu:
            self.relu = ACTIVATE[act](inplace=True)

    def forward(self, x):
        if self.use_relu:
            return self.relu(self.bn(self.conv(x)))
        else:
            return self.bn(self.conv(x))


class depth_wise(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, act='leaky'):
        super(depth_wise, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = ACTIVATE[act](inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class PEP(nn.Module):
    """
    This is yolo_nano PEP module
    """

    def __init__(self, in_dim, mid_dim, out_dim, stride=1, groups=1, ratios=2, act='leaky'):
        super(PEP, self).__init__()
        self.conv1X1_0 = conv1x1(in_dim, mid_dim, stride, groups, act=act)
        self.conv1X1_1 = conv1x1(mid_dim, mid_dim * ratios, stride, groups, act=act)
        self.depth_wise = depth_wise(mid_dim * ratios, mid_dim * ratios, 1, act=act)
        self.conv1X1_2 = conv1x1(mid_dim * ratios, out_dim, stride, groups, act=act, use_relu=False)
        self.relu = ACTIVATE[act](inplace=True)
        if stride != 1 or in_dim != out_dim:
            self.downsample = conv1x1(in_dim, out_dim, stride=stride, groups=groups, act=act, use_relu=False)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1X1_0(x)
        out = self.conv1X1_1(out)
        out = self.depth_wise(out)
        out = self.conv1X1_2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(identity + out)


class FCA(nn.Module):
    """
    Module structure FCA some like
    """

    def __init__(self, channels, reduce_channels):
        super(FCA, self).__init__()
        self.channels = channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, reduce_channels, bias=False), nn.ReLU6(inplace=True), nn.Linear(reduce_channels, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x).view(b, c)
        out = self.fc(out).view(b, c, 1, 1)
        out = x * out.expand_as(x)
        return out


class EP(nn.Module):

    def __init__(self, input_channels, output_channels, stride=1, groups=1, act='leaky'):
        super(EP, self).__init__()
        self.conv1x1_0 = conv1x1(input_channels, output_channels, 1, groups, act=act)
        self.depth_wise = depth_wise(output_channels, output_channels, stride, act=act)
        self.conv1x1_1 = conv1x1(output_channels, output_channels, stride=1, groups=groups, act=act, use_relu=False)
        if stride != 1 or input_channels != output_channels:
            self.downsample = conv1x1(input_channels, output_channels, stride=stride, groups=groups, act=act, use_relu=False)
        else:
            self.downsample = None
        self.relu = ACTIVATE[act](inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1x1_0(x)
        out = self.depth_wise(out)
        out = self.conv1x1_1(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(identity + out)


ARCHITECTURE = {'layer3': [['EP', 150, 325, 2], ['PEP', 325, 132, 325], ['PEP', 325, 124, 325], ['PEP', 325, 141, 325], ['PEP', 325, 140, 325], ['PEP', 325, 137, 325], ['PEP', 325, 135, 325], ['PEP', 325, 133, 325], ['PEP', 325, 140, 325]], 'layer4': [['EP', 325, 545, 2], ['PEP', 545, 276, 545], ['conv1x1', 545, 230], ['EP', 230, 489, 1], ['PEP', 489, 213, 469], ['conv1x1', 469, 189]]}


YOLO_ARCH = {'small': [(116, 90), (156, 198), (373, 326)], 'middle': [(30, 61), (62, 45), (59, 119)], 'large': [(10, 13), (16, 30), (33, 23)]}


class YoloNano(nn.Module):
    """
    Paper Structure Arch
    return three scale feature,int the paper :(52)4,(26)2,(13)1.
    each channel in here is only 75. for voc 2007 because: (num_class+5)*anchor because voc has 20 classes include background so the
    channel in here is 75
    """

    def __init__(self, num_class=20, num_anchor=3, img_size=416):
        __FUNC = {'EP': EP, 'PEP': PEP, 'conv1x1': conv1x1}
        super(YoloNano, self).__init__()
        self.num_class = num_class
        self.num_anchor = num_anchor
        self.img_size = img_size
        self.out_channel = (num_class + 5) * num_anchor
        self.seen = 0
        self.layer0 = nn.Sequential(conv3x3(3, 12, 1), conv3x3(12, 24, 2))
        self.layer1 = nn.Sequential(PEP(24, 7, 24), EP(24, 70, 2), PEP(70, 25, 70), PEP(70, 24, 70), EP(70, 150, 2), PEP(150, 56, 150), conv1x1(150, 150, 1, 1, 1, use_relu=True))
        self.attention = FCA(150, 8)
        self.layer2 = nn.Sequential(PEP(150, 73, 150), PEP(150, 71, 150), PEP(150, 75, 150))
        layer3 = []
        for e in ARCHITECTURE['layer3']:
            layer3.append(__FUNC[e[0]](*e[1:]))
        self.layer3 = nn.Sequential(*layer3)
        layer4 = []
        for e in ARCHITECTURE['layer4']:
            layer4.append(__FUNC[e[0]](*e[1:]))
        self.layer4 = nn.Sequential(*layer4)
        self.layer5 = nn.Sequential(PEP(430, 113, 325), PEP(325, 99, 207), conv1x1(207, 98, use_relu=True))
        self.compress = conv1x1(189, 105, use_relu=True)
        self.compress2 = conv1x1(98, 47, use_relu=True)
        self.scale_4 = nn.Sequential(PEP(197, 58, 122), PEP(122, 52, 87), PEP(87, 47, 93), nn.Conv2d(93, self.out_channel, kernel_size=1, stride=1, padding=0, bias=True))
        self.scale_2 = nn.Sequential(EP(98, 183, 1), nn.Conv2d(183, self.out_channel, kernel_size=1, stride=1, padding=0, bias=True))
        self.scale_1 = nn.Sequential(EP(189, 462, 1), nn.Conv2d(462, self.out_channel, kernel_size=1, stride=1, padding=0, bias=True))
        self.yolo0 = YOLOLayer(YOLO_ARCH['small'], self.num_class, self.img_size)
        self.yolo1 = YOLOLayer(YOLO_ARCH['middle'], self.num_class, self.img_size)
        self.yolo2 = YOLOLayer(YOLO_ARCH['large'], self.num_class, self.img_size)
        self.yolo_layers = [self.yolo0, self.yolo1, self.yolo2]

    def forward(self, x, targets=None, img_scores=None, gt_mix_index=None):
        img_dim = x.shape[2]
        loss = 0
        yolo_outputs = []
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.attention(x)
        x_1 = self.layer2(x)
        x_2 = self.layer3(x_1)
        x_3 = self.layer4(x_2)
        x = self.compress(x_3)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x_2], dim=1)
        x_4 = self.layer5(x)
        x = self.compress2(x_4)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, x_1], dim=1)
        x_scale_4 = self.scale_4(x)
        x_scale_2 = self.scale_2(x_4)
        x_scale_1 = self.scale_1(x_3)
        layer_0_x, layer_loss = self.yolo0(x_scale_1, targets, img_dim, img_scores=img_scores, gt_mix_index=gt_mix_index)
        loss += layer_loss
        yolo_outputs.append(layer_0_x)
        layer_1_x, layer_loss = self.yolo1(x_scale_2, targets, img_dim, img_scores=img_scores, gt_mix_index=gt_mix_index)
        loss += layer_loss
        yolo_outputs.append(layer_1_x)
        layer_2_x, layer_loss = self.yolo2(x_scale_4, targets, img_dim, img_scores=img_scores, gt_mix_index=gt_mix_index)
        loss += layer_loss
        yolo_outputs.append(layer_2_x)
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (EP,
     lambda: ([], {'input_channels': 4, 'output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FCA,
     lambda: ([], {'channels': 4, 'reduce_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PEP,
     lambda: ([], {'in_dim': 4, 'mid_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Upsample,
     lambda: ([], {'scale_factor': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (YoloNano,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (conv1x1,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (conv3x3,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (depth_wise,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_lingtengqiu_Yolo_Nano(_paritybench_base):
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

