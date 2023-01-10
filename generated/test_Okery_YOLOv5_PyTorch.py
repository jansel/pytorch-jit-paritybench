import sys
_module = sys.modules[__name__]
del sys
train = _module
yolo = _module
__mess = _module
datasets = _module
coco_dataset = _module
coco_eval = _module
dali = _module
generalized_dataset = _module
transforms = _module
utils = _module
voc_dataset = _module
distributed = _module
engine = _module
gpu = _module
model = _module
backbone = _module
backbone_utils = _module
darknet = _module
path_aggregation_network = _module
utils = _module
box_ops = _module
head = _module
transform = _module
yolo = _module
utils = _module
visualize = _module

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


import re


import time


import torch


from collections import OrderedDict


import copy


from torchvision import transforms


import random


import numpy as np


import torch.nn.functional as F


from collections import defaultdict


import torch.distributed as dist


from torch import nn


import matplotlib.pyplot as plt


from matplotlib import patches


class BackboneWithFPN(nn.Module):

    def __init__(self, backbone, fpn):
        super().__init__()
        self.body = backbone
        self.fpn = fpn

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


class MishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.mul(torch.tanh(F.softplus(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        fx = F.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))


class Mish(nn.Module):

    def forward(self, x):
        return MishImplementation.apply(x)


bn_eps = 0.0001


bn_momentum = 0.03


def fuse_conv_and_bn(conv, bn):
    conv_w, conv_b = conv.weight, conv.bias
    bn_w, bn_b = bn.weight, bn.bias
    bn_rm, bn_rv, bn_eps = bn.running_mean, bn.running_var, bn.eps
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
    fconv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(-1, 1, 1, 1)
    fconv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    fused_conv = copy.deepcopy(conv)
    fused_conv.weight = nn.Parameter(fconv_w)
    fused_conv.bias = nn.Parameter(fconv_b)
    return fused_conv


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, acti='leaky'):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum)
        if acti == 'relu':
            self.acti = nn.ReLU(inplace=True)
        elif acti == 'leaky':
            self.acti = nn.LeakyReLU(0.1, inplace=True)
        elif acti == 'mish':
            self.acti = Mish()
        self.fused = False

    def forward(self, x):
        if not self.training and self.fused:
            return self.acti(self.fused_conv[0](x))
        else:
            return self.acti(self.bn(self.conv(x)))

    def fuse(self):
        self.fused = True
        self.fused_conv = fuse_conv_and_bn(self.conv, self.bn),


class SpatialPyramidPooling(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = Conv(in_channels, mid_channels, 1)
        self.conv2 = Conv(4 * mid_channels, out_channels, 1)
        self.pool1 = nn.MaxPool2d(5, 1, 2)
        self.pool2 = nn.MaxPool2d(9, 1, 4)
        self.pool3 = nn.MaxPool2d(13, 1, 6)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        out = torch.cat((x, x1, x2, x3), dim=1)
        out = self.conv2(out)
        return out


class Focus(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = Conv(4 * in_channels, out_channels, kernel_size)
        self.out_channels = out_channels

    def forward(self, x):
        concat = torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1)
        return self.conv(concat)


class FusionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, fusion):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, 1)
        self.conv2 = Conv(out_channels, out_channels, 3)
        self.fusion = fusion and in_channels == out_channels

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.fusion:
            out = out + x
        return out


class ConcatBlock(nn.Module):

    def __init__(self, in_channels, out_channels, layer, fusion):
        super().__init__()
        mid_channels = out_channels // 2
        self.part1 = nn.Sequential(Conv(in_channels, mid_channels, 1), nn.Sequential(*[FusionBlock(mid_channels, mid_channels, fusion) for _ in range(layer)]), nn.Conv2d(mid_channels, mid_channels, 1, bias=False))
        self.part2 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.tail = nn.Sequential(nn.BatchNorm2d(2 * mid_channels, eps=bn_eps, momentum=bn_momentum), nn.LeakyReLU(0.1, inplace=True))
        self.conv = Conv(2 * mid_channels, out_channels, 1)

    def forward(self, x):
        x1 = self.part1(x)
        x2 = self.part2(x)
        out = self.conv(self.tail(torch.cat((x1, x2), dim=1)))
        return out


class CSPDarknet(nn.Sequential):

    def __init__(self, out_channels_list, layers):
        assert len(layers) + 2 == len(out_channels_list), 'len(layers) != len(out_channels_list)'
        d = OrderedDict()
        d['layer0'] = Focus(3, out_channels_list[0], 3)
        for i, ch in enumerate(out_channels_list[1:]):
            in_channels = out_channels_list[i]
            name = 'layer{}'.format(i + 1)
            d[name] = nn.Sequential(Conv(in_channels, ch, 3, 2))
            if i < len(out_channels_list) - 2:
                d[name].add_module('concat', ConcatBlock(ch, ch, layers[i], True))
            else:
                d[name].add_module('spp', SpatialPyramidPooling(ch, ch))
        super().__init__(d)


class PathAggregationNetwork(nn.Module):

    def __init__(self, in_channels_list, depth):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        self.outer_blocks = nn.ModuleList()
        for i, ch in enumerate(in_channels_list):
            self.inner_blocks.append(ConcatBlock(2 * ch if i < 2 else in_channels_list[-1], ch, depth, False))
            if i > 0:
                in_channels = in_channels_list[i - 1]
                self.layer_blocks.append(Conv(ch, in_channels, 1))
                self.upsample_blocks.append(nn.Upsample(scale_factor=2))
                self.downsample_blocks.append(Conv(in_channels, in_channels, 3, 2))
                self.outer_blocks.append(ConcatBlock(ch, ch, depth, False))

    def forward(self, x):
        results = []
        last_inner = self.inner_blocks[-1](x[-1])
        results.append(self.layer_blocks[-1](last_inner))
        for i in range(len(x) - 2, -1, -1):
            inner_top_down = self.upsample_blocks[i](results[0])
            last_inner = self.inner_blocks[i](torch.cat((inner_top_down, x[i]), dim=1))
            results.insert(0, last_inner if i == 0 else self.layer_blocks[i - 1](last_inner))
        for i in range(len(x) - 1):
            outer_bottom_up = self.downsample_blocks[i](results[i])
            layer_result = results[i + 1]
            results[i + 1] = self.outer_blocks[i](torch.cat((outer_bottom_up, layer_result), dim=1))
        return results


class IntermediateLayerGetter(nn.ModuleDict):

    def __init__(self, model, return_layers):
        if not return_layers.issubset([name for name, _ in model.named_children()]):
            raise ValueError('return_layers are not present in model')
        self.return_layers = return_layers
        layers = OrderedDict()
        n = 0
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                n += 1
            if n == len(return_layers):
                break
        super().__init__(layers)

    def forward(self, x):
        outputs = []
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                outputs.append(x)
        return outputs


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False))

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        Mc = torch.sigmoid(avg_out + max_out)
        return Mc


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = kernel_size, kernel_size
        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        out = torch.cat((avg_out, max_out), dim=1)
        Ms = torch.sigmoid(self.conv(out))
        return Ms


class ConvBlockAttention(nn.Module):

    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super().__init__()
        self.cam = ChannelAttention(in_channels, ratio)
        self.sam = SpatialAttention(kernel_size)

    def forward(self, x):
        Mc = self.cam(x)
        x = x * Mc
        Ms = self.sam(x)
        x = x * Ms
        return x


class Head(nn.Module):

    def __init__(self, predictor, anchors, strides, match_thresh, giou_ratio, loss_weights, score_thresh, nms_thresh, detections):
        super().__init__()
        self.predictor = predictor
        self.register_buffer('anchors', torch.Tensor(anchors))
        self.strides = strides
        self.match_thresh = match_thresh
        self.giou_ratio = giou_ratio
        self.loss_weights = loss_weights
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections = detections
        self.merge = False
        self.eval_with_loss = False

    def forward(self, features, targets, image_shapes=None, scale_factors=None, max_size=None):
        preds = self.predictor(features)
        if self.training:
            losses = self.compute_loss(preds, targets)
            return losses
        else:
            losses = {}
            if self.eval_with_loss:
                losses = self.compute_loss(preds, targets)
            results = self.inference(preds, image_shapes, scale_factors, max_size)
            return results, losses

    def compute_loss(self, preds, targets):
        dtype = preds[0].dtype
        image_ids = torch.cat([torch.full_like(tgt['labels'], i) for i, tgt in enumerate(targets)])
        gt_labels = torch.cat([tgt['labels'] for tgt in targets])
        gt_boxes = torch.cat([tgt['boxes'] for tgt in targets])
        gt_boxes = box_ops.xyxy2cxcywh(gt_boxes)
        losses = {'loss_box': gt_boxes.new_tensor(0), 'loss_obj': gt_boxes.new_tensor(0), 'loss_cls': gt_boxes.new_tensor(0)}
        for pred, stride, wh in zip(preds, self.strides, self.anchors):
            anchor_id, gt_id = box_ops.size_matched_idx(wh, gt_boxes[:, 2:], self.match_thresh)
            gt_object = torch.zeros_like(pred[..., 4])
            if len(anchor_id) > 0:
                gt_box_xy = gt_boxes[:, :2][gt_id]
                ids, grid_xy = box_ops.assign_targets_to_proposals(gt_box_xy / stride, pred.shape[1:3])
                anchor_id, gt_id = anchor_id[ids], gt_id[ids]
                image_id = image_ids[gt_id]
                pred_level = pred[image_id, grid_xy[:, 1], grid_xy[:, 0], anchor_id]
                xy = 2 * torch.sigmoid(pred_level[:, :2]) - 0.5 + grid_xy
                wh = 4 * torch.sigmoid(pred_level[:, 2:4]) ** 2 * wh[anchor_id] / stride
                box_grid = torch.cat((xy, wh), dim=1)
                giou = box_ops.box_giou(box_grid, gt_boxes[gt_id] / stride)
                losses['loss_box'] += (1 - giou).mean()
                gt_object[image_id, grid_xy[:, 1], grid_xy[:, 0], anchor_id] = self.giou_ratio * giou.detach().clamp(0) + (1 - self.giou_ratio)
                gt_label = torch.zeros_like(pred_level[..., 5:])
                gt_label[range(len(gt_id)), gt_labels[gt_id]] = 1
                losses['loss_cls'] += F.binary_cross_entropy_with_logits(pred_level[..., 5:], gt_label)
            losses['loss_obj'] += F.binary_cross_entropy_with_logits(pred[..., 4], gt_object)
        losses = {k: (v * self.loss_weights[k]) for k, v in losses.items()}
        return losses

    def inference(self, preds, image_shapes, scale_factors, max_size):
        ids, ps, boxes = [], [], []
        for pred, stride, wh in zip(preds, self.strides, self.anchors):
            pred = torch.sigmoid(pred)
            n, y, x, a = torch.where(pred[..., 4] > self.score_thresh)
            p = pred[n, y, x, a]
            xy = torch.stack((x, y), dim=1)
            xy = (2 * p[:, :2] - 0.5 + xy) * stride
            wh = 4 * p[:, 2:4] ** 2 * wh[a]
            box = torch.cat((xy, wh), dim=1)
            ids.append(n)
            ps.append(p)
            boxes.append(box)
        ids = torch.cat(ids)
        ps = torch.cat(ps)
        boxes = torch.cat(boxes)
        boxes = box_ops.cxcywh2xyxy(boxes)
        logits = ps[:, [4]] * ps[:, 5:]
        indices, labels = torch.where(logits > self.score_thresh)
        ids, boxes, scores = ids[indices], boxes[indices], logits[indices, labels]
        results = []
        for i, im_s in enumerate(image_shapes):
            keep = torch.where(ids == i)[0]
            box, label, score = boxes[keep], labels[keep], scores[keep]
            if len(box) > 0:
                box[:, 0].clamp_(0, im_s[1])
                box[:, 1].clamp_(0, im_s[0])
                box[:, 2].clamp_(0, im_s[1])
                box[:, 3].clamp_(0, im_s[0])
                keep = box_ops.batched_nms(box, score, label, self.nms_thresh, max_size)
                keep = keep[:self.detections]
                nms_box, nms_label = box[keep], label[keep]
                if self.merge:
                    mask = nms_label[:, None] == label[None]
                    iou = box_ops.box_iou(nms_box, box) * mask > self.nms_thresh
                    weights = iou * score[None]
                    nms_box = torch.mm(weights, box) / weights.sum(1, keepdim=True)
                box, label, score = nms_box / scale_factors[i], nms_label, score[keep]
            results.append(dict(boxes=box, labels=label, scores=score))
        return results


def sort_images(shapes, out, dim):
    shapes.sort(key=lambda x: x[dim])
    out.append(shapes.pop()[2])
    if dim == 0:
        out.append(shapes.pop()[2])
        out.append(shapes.pop(1)[2])
    else:
        out.append(shapes.pop(1)[2])
        out.append(shapes.pop()[2])
    out.append(shapes.pop(0)[2])
    if shapes:
        sort_images(shapes, out, dim)


def mosaic_augment(images, targets):
    assert len(images) % 4 == 0, 'mosaic augmentation: len(images) % 4 != 0'
    shapes = [(img.shape[1], img.shape[2], i) for i, img in enumerate(images)]
    ratios = [int(h >= w) for h, w, _ in shapes]
    dim = int(sum(ratios) >= len(ratios) * 0.5)
    order = []
    sort_images(shapes, order, dim)
    new_images, new_targets = [], []
    for i in range(len(order) // 4):
        hs, ws = zip(*[images[o].shape[-2:] for o in order[4 * i:4 * (i + 1)]])
        tl_y, br_y = max(hs[0], hs[1]), max(hs[2], hs[3])
        tl_x, br_x = max(ws[0], ws[2]), max(ws[1], ws[3])
        merged_image = images[0].new_zeros((3, tl_y + br_y, tl_x + br_x))
        for j in range(4):
            index = order[4 * i + j]
            img = images[index]
            box = targets[index]['boxes']
            h, w = img.shape[-2:]
            x1, y1, x2, y2 = tl_x, tl_y, tl_x, tl_y
            if j == 0:
                x1 -= w
                y1 -= h
            elif j == 1:
                x2 += w
                y1 -= h
            elif j == 2:
                x1 -= w
                y2 += h
            elif j == 3:
                x2 += w
                y2 += h
            merged_image[:, y1:y2, x1:x2].copy_(img)
            box[:, [0, 2]] += x1
            box[:, [1, 3]] += y1
        boxes = torch.cat([targets[o]['boxes'] for o in order[4 * i:4 * (i + 1)]])
        labels = torch.cat([targets[o]['labels'] for o in order[4 * i:4 * (i + 1)]])
        new_images.append(merged_image)
        new_targets.append(dict(boxes=boxes, labels=labels))
    return new_images, new_targets


class Transformer(nn.Module):

    def __init__(self, min_size, max_size, stride=32):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.stride = stride
        self.flip_prob = 0.5
        self.mosaic = False

    def forward(self, images, targets):
        if targets is None:
            transformed = [self.transforms(img, targets) for img in images]
        else:
            targets = copy.deepcopy(targets)
            transformed = [self.transforms(img, tgt) for img, tgt in zip(images, targets)]
        images, targets, scale_factors = zip(*transformed)
        image_shapes = None
        if self.training:
            if self.mosaic:
                images, targets = mosaic_augment(images, targets)
        else:
            image_shapes = [img.shape[1:] for img in images]
        images = self.batch_images(images)
        return images, targets, scale_factors, image_shapes

    def transforms(self, image, target):
        image, target, scale_factor = self.resize(image, target)
        if self.training:
            if random.random() < self.flip_prob:
                image, target['boxes'] = self.horizontal_flip(image, target['boxes'])
        return image, target, scale_factor

    def horizontal_flip(self, image, boxes):
        w = image.shape[2]
        image = image.flip(2)
        tmp = boxes[:, 0] + 0
        boxes[:, 0] = w - boxes[:, 2]
        boxes[:, 2] = w - tmp
        return image, boxes

    def resize(self, image, target):
        orig_image_shape = image.shape[1:]
        min_size = min(orig_image_shape)
        max_size = max(orig_image_shape)
        scale_factor = min(self.min_size / min_size, self.max_size / max_size)
        if scale_factor != 1:
            size = [round(s * scale_factor) for s in orig_image_shape]
            image = F.interpolate(image[None], size=size, mode='bilinear', align_corners=False)[0]
            if target is not None:
                box = target['boxes']
                box[:, [0, 2]] *= size[1] / orig_image_shape[1]
                box[:, [1, 3]] *= size[0] / orig_image_shape[0]
        return image, target, scale_factor

    def batch_images(self, images):
        max_size = tuple(max(s) for s in zip(*(img.shape[1:] for img in images)))
        batch_size = tuple(math.ceil(m / self.stride) * self.stride for m in max_size)
        batch_shape = (len(images), 3) + batch_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[:, :img.shape[1], :img.shape[2]].copy_(img)
        return batched_imgs


class Predictor(nn.Module):

    def __init__(self, in_channels_list, num_anchors, num_classes, strides):
        super().__init__()
        self.num_outputs = num_classes + 5
        self.mlp = nn.ModuleList()
        for in_channels, n in zip(in_channels_list, num_anchors):
            out_channels = n * self.num_outputs
            self.mlp.append(nn.Conv2d(in_channels, out_channels, 1))
        for m, n, s in zip(self.mlp, num_anchors, strides):
            b = m.bias.detach().view(n, -1)
            b[:, 4] += math.log(8 / (416 / s) ** 2)
            b[:, 5:] += math.log(0.6 / (num_classes - 0.99))
            m.bias = nn.Parameter(b.view(-1))

    def forward(self, x):
        N = x[0].shape[0]
        L = self.num_outputs
        preds = []
        for i in range(len(x)):
            h, w = x[i].shape[-2:]
            pred = self.mlp[i](x[i])
            pred = pred.permute(0, 2, 3, 1).reshape(N, h, w, -1, L)
            preds.append(pred)
        return preds


def darknet_pan_backbone(depth_multiple, width_multiple):
    out_channels_list = [round(width_multiple * x) for x in [64, 128, 256, 512, 1024]]
    layers = [max(round(depth_multiple * x), 1) for x in [3, 9, 9]]
    model = CSPDarknet(out_channels_list, layers)
    return_layers = {'layer2', 'layer3', 'layer4'}
    backbone = IntermediateLayerGetter(model, return_layers)
    backbone.out_channels_list = out_channels_list[2:]
    depth = max(round(3 * depth_multiple), 1)
    fpn = PathAggregationNetwork(out_channels_list[2:], depth)
    return BackboneWithFPN(backbone, fpn)


class YOLOv5(nn.Module):

    def __init__(self, num_classes, model_size=(0.33, 0.5), match_thresh=4, giou_ratio=1, img_sizes=(320, 416), score_thresh=0.1, nms_thresh=0.6, detections=100):
        super().__init__()
        anchors1 = [[[10, 13], [16, 30], [33, 23]], [[30, 61], [62, 45], [59, 119]], [[116, 90], [156, 198], [373, 326]]]
        anchors = [[[6.1, 8.1], [20.6, 12.6], [11.2, 23.7]], [[36.2, 26.8], [25.9, 57.2], [57.8, 47.9]], [[122.1, 78.3], [73.7, 143.8], [236.1, 213.1]]]
        loss_weights = {'loss_box': 0.05, 'loss_obj': 1.0, 'loss_cls': 0.5}
        self.backbone = darknet_pan_backbone(depth_multiple=model_size[0], width_multiple=model_size[1])
        in_channels_list = self.backbone.body.out_channels_list
        strides = 8, 16, 32
        num_anchors = [len(s) for s in anchors]
        predictor = Predictor(in_channels_list, num_anchors, num_classes, strides)
        self.head = Head(predictor, anchors, strides, match_thresh, giou_ratio, loss_weights, score_thresh, nms_thresh, detections)
        if isinstance(img_sizes, int):
            img_sizes = img_sizes, img_sizes
        self.transformer = Transformer(min_size=img_sizes[0], max_size=img_sizes[1], stride=max(strides))

    def forward(self, images, targets=None):
        images, targets, scale_factors, image_shapes = self.transformer(images, targets)
        features = self.backbone(images)
        if self.training:
            losses = self.head(features, targets)
            return losses
        else:
            max_size = max(images.shape[2:])
            results, losses = self.head(features, targets, image_shapes, scale_factors, max_size)
            return results, losses

    def fuse(self):
        for m in self.modules():
            if hasattr(m, 'fused'):
                m.fuse()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BackboneWithFPN,
     lambda: ([], {'backbone': _mock_layer(), 'fpn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConcatBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'layer': 1, 'fusion': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Focus,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FusionBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'fusion': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SpatialAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialPyramidPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Okery_YOLOv5_PyTorch(_paritybench_base):
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

