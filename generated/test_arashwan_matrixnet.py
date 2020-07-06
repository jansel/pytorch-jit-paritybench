import sys
_module = sys.modules[__name__]
del sys
config = _module
db = _module
base = _module
coco = _module
datasets = _module
detection = _module
external = _module
setup = _module
MatrixNetAnchors = _module
MatrixNetCorners = _module
models = _module
matrixnet = _module
py_utils = _module
data_parallel = _module
loss_utils = _module
scatter_gather = _module
utils = _module
resnet_features = _module
nnet = _module
py_factory = _module
sample = _module
coco = _module
test = _module
coco = _module
train = _module
image = _module
tqdm = _module

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


import math


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


from torch.nn.modules import Module


from torch.nn.parallel.scatter_gather import gather


from torch.nn.parallel.replicate import replicate


from torch.nn.parallel.parallel_apply import parallel_apply


from torch.nn.parallel._functions import Scatter


from torch.nn.parallel._functions import Gather


import torch.utils.model_zoo as model_zoo


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import Bottleneck


from torchvision.models.resnet import ResNet


import numpy as np


import random


import string


from random import randrange


import queue


from torch.multiprocessing import Process


from torch.multiprocessing import Queue


from torch.multiprocessing import Pool


def init_conv_weights(layer, weights_std=0.01, bias=0):
    """
    RetinaNet's layer initialization
    :layer
    :
    """
    nn.init.normal_(layer.weight, std=weights_std)
    nn.init.constant_(layer.bias, val=bias)
    return layer


def conv3x3(in_channels, out_channels, **kwargs):
    """Return a 3x3 convolutional layer with RetinaNet's weight and bias initialization"""
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, **kwargs)
    layer = init_conv_weights(layer)
    return layer


class SubNet(nn.Module):

    def __init__(self, mode, classes=80, depth=4, base_activation=F.relu, output_activation=F.sigmoid):
        super(SubNet, self).__init__()
        self.classes = classes
        self.depth = depth
        self.base_activation = base_activation
        self.output_activation = output_activation
        self.subnet_base = nn.ModuleList([conv3x3(256, 256, padding=1) for _ in range(depth)])
        if mode == 'corners':
            self.subnet_output = conv3x3(256, 2, padding=1)
        if mode == 'centers':
            self.subnet_output = conv3x3(256, 2, padding=1)
        elif mode == 'classes':
            self.subnet_output = conv3x3(256, self.classes, padding=1)

    def forward(self, x):
        for layer in self.subnet_base:
            x = self.base_activation(layer(x))
        x = self.subnet_output(x)
        return x


def conv1x1(in_channels, out_channels, **kwargs):
    """Return a 1x1 convolutional layer with RetinaNet's weight and bias initialization"""
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)
    layer = init_conv_weights(layer)
    return layer


class MatrixNet(nn.Module):

    def __init__(self, resnet, layers):
        super(MatrixNet, self).__init__()
        self.resnet = resnet
        self.incidence = {(33): [34, 43, 44, 22], (34): [35], (43): [53], (44): [45, 55, 54], (22): [11, 32, 23], (11): [12, 21], (32): [42], (23): [24], (12): [13], (21): [31], (42): [52], (24): [25], (13): [14], (31): [41], (14): [15], (41): [51]}
        self.visited = set()
        self.layers = layers
        self.keeps = set()
        for i, l in enumerate(self.layers):
            for j, e in enumerate(l):
                if e != -1:
                    self.keeps.add((j + 1) * 10 + (i + 1))

        def _bfs(graph, start, end):
            queue = []
            queue.append([start])
            while queue:
                path = queue.pop(0)
                node = path[-1]
                if node == end:
                    return path
                for n in graph.get(node, []):
                    new_path = list(path)
                    new_path.append(n)
                    queue.append(new_path)
        _keeps = self.keeps.copy()
        while _keeps:
            node = _keeps.pop()
            vs = set(_bfs(self.incidence, 33, node))
            self.visited = vs | self.visited
            _keeps = _keeps - self.visited
        self.pyramid_transformation_3 = conv1x1(512, 256)
        self.pyramid_transformation_4 = conv1x1(1024, 256)
        self.pyramid_transformation_5 = conv1x1(2048, 256)
        self.pyramid_transformation_6 = conv3x3(2048, 256, padding=1, stride=2)
        self.pyramid_transformation_7 = conv3x3(256, 256, padding=1, stride=2)
        self.upsample_transform_1 = conv3x3(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3(256, 256, padding=1)
        self.downsample_transformation_12 = conv3x3(256, 256, padding=1, stride=(1, 2))
        self.downsample_transformation_21 = conv3x3(256, 256, padding=1, stride=(2, 1))

    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        height, width = scaled_feature.size()[2:]
        return F.interpolate(original_feature, scale_factor=scale_factor)[:, :, :height, :width]

    def forward(self, x):
        _, resnet_feature_3, resnet_feature_4, resnet_feature_5 = self.resnet(x)
        _dict = {}
        if 44 in self.visited:
            _dict[44] = self.pyramid_transformation_6(resnet_feature_5)
        if 55 in self.visited:
            _dict[55] = self.pyramid_transformation_7(F.relu(_dict[44]))
        if 33 in self.visited:
            _dict[33] = self.pyramid_transformation_5(resnet_feature_5)
        if 22 in self.visited:
            _dict[22] = self.pyramid_transformation_4(resnet_feature_4)
        if 33 in self.visited and 22 in self.visited:
            upsampled_feature_5 = self._upsample(_dict[33], _dict[22])
        if 22 in self.visited:
            _dict[22] = self.upsample_transform_1(torch.add(upsampled_feature_5, _dict[22]))
        if 11 in self.visited:
            _dict[11] = self.pyramid_transformation_3(resnet_feature_3)
        if 11 in self.visited and 22 in self.visited:
            upsampled_feature_4 = self._upsample(_dict[22], _dict[11])
        if 11 in self.visited:
            _dict[11] = self.upsample_transform_2(torch.add(upsampled_feature_4, _dict[11]))
        if 12 in self.visited:
            _dict[12] = self.downsample_transformation_12(_dict[11])
        if 13 in self.visited:
            _dict[13] = self.downsample_transformation_12(_dict[12])
        if 14 in self.visited:
            _dict[14] = self.downsample_transformation_12(_dict[13])
        if 15 in self.visited:
            _dict[15] = self.downsamole_transformation_12(_dict[14])
        if 21 in self.visited:
            _dict[21] = self.downsample_transformation_21(_dict[11])
        if 31 in self.visited:
            _dict[31] = self.downsample_transformation_21(_dict[21])
        if 41 in self.visited:
            _dict[41] = self.downsample_transformation_21(_dict[31])
        if 51 in self.visited:
            _dict[51] = self.downsample_transformation_21(_dict[41])
        if 23 in self.visited:
            _dict[23] = self.downsample_transformation_12(_dict[22])
        if 24 in self.visited:
            _dict[24] = self.downsample_transformation_12(_dict[23])
        if 25 in self.visited:
            _dict[25] = self.downsample_transformation_12(_dict[24])
        if 32 in self.visited:
            _dict[32] = self.downsample_transformation_21(_dict[22])
        if 42 in self.visited:
            _dict[42] = self.downsample_transformation_21(_dict[32])
        if 52 in self.visited:
            _dict[52] = self.downsample_transformation_21(_dict[42])
        if 34 in self.visited:
            _dict[34] = self.downsample_transformation_12(_dict[33])
        if 35 in self.visited:
            _dict[35] = self.downsample_transformation_12(_dict[34])
        if 43 in self.visited:
            _dict[43] = self.downsample_transformation_21(_dict[33])
        if 53 in self.visited:
            _dict[53] = self.downsample_transformation_21(_dict[43])
        if 45 in self.visited:
            _dict[45] = self.downsample_transformation_12(_dict[44])
        if 54 in self.visited:
            _dict[54] = self.downsample_transformation_21(_dict[44])
        order_keeps = {(i % 10 * 10 + i // 10): i for i in self.keeps}
        return [_dict[order_keeps[i]] for i in sorted(order_keeps)]


def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=0.0001, max=1 - 0.0001)
    return x


class BottleneckFeatures(Bottleneck):
    """
    Bottleneck that returns its last conv layer features.
    """

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        conv3_rep = out
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out, conv3_rep


class ResNetFeatures(ResNet):
    """
    A ResNet that returns features instead of classification.
    """

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x, c2 = self.layer1(x)
        x, c3 = self.layer2(x)
        x, c4 = self.layer3(x)
        x, c5 = self.layer4(x)
        return c2, c3, c4, c5


model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth', 'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth', 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth', 'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth', 'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth'}


def resnet101_features(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFeatures(BottleneckFeatures, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152_features(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFeatures(BottleneckFeatures, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnet50_features(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFeatures(BottleneckFeatures, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnext101_32x8d(pretrained=False, **kwargs):
    """ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    model = ResNetFeatures(BottleneckFeatures, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnext101_32x8d']))
    return model


class MatrixNetAnchors(nn.Module):

    def __init__(self, classes, resnet, layers):
        super(MatrixNetAnchors, self).__init__()
        self.classes = classes
        self.resnet = resnet
        if self.resnet == 'resnext101_32x8d':
            _resnet = resnext101_32x8d(pretrained=True)
        elif self.resnet == 'resnet101':
            _resnet = resnet101_features(pretrained=True)
        elif self.resnet == 'resnet50':
            _resnet = resnet50_features(pretrained=True)
        elif self.resnet == 'resnet152':
            _resnet = resnet152_features(pretrained=True)
        try:
            self.matrix_net = MatrixNet(_resnet, layers)
        except:
            None
            sys.exit()
        self.subnet_tl_corners_regr = SubNet(mode='tl_corners')
        self.subnet_br_corners_regr = SubNet(mode='br_corners')
        self.subnet_anchors_heats = SubNet(mode='classes')

    def forward(self, x):
        features = self.matrix_net(x)
        anchors_tl_corners_regr = [self.subnet_tl_corners_regr(feature) for feature in features]
        anchors_br_corners_regr = [self.subnet_br_corners_regr(feature) for feature in features]
        anchors_heatmaps = [_sigmoid(self.subnet_anchors_heats(feature)) for feature in features]
        return anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr


class MatrixNetCorners(nn.Module):

    def __init__(self, classes, resnet, layers):
        super(MatrixNetCorners, self).__init__()
        self.classes = classes
        self.resnet = resnet
        if self.resnet == 'resnext101_32x8d':
            _resnet = resnext101_32x8d(pretrained=True)
        elif self.resnet == 'resnet101':
            _resnet = resnet101_features(pretrained=True)
        elif self.resnet == 'resnet50':
            _resnet = resnet50_features(pretrained=True)
        elif self.resnet == 'resnet152':
            _resnet = resnet152_features(pretrained=True)
        try:
            self.feature_pyramid = MatrixNet(_resnet, layers)
        except:
            None
            sys.exit()
        self.subnet_tl_corners_regr = SubNet(mode='corners')
        self.subnet_tl_centers_regr = SubNet(mode='centers')
        self.subnet_br_corners_regr = SubNet(mode='corners')
        self.subnet_br_centers_regr = SubNet(mode='centers')
        self.subnet_tl_heats = SubNet(mode='classes')
        self.subnet_br_heats = SubNet(mode='classes')

    def forward(self, x):
        features = self.feature_pyramid(x)
        tl_corners_regr = [self.subnet_tl_corners_regr(feature) for feature in features]
        tl_centers_regr = [F.relu(self.subnet_tl_centers_regr(feature)) for feature in features]
        br_corners_regr = [self.subnet_br_corners_regr(feature) for feature in features]
        br_centers_regr = [F.relu(self.subnet_br_centers_regr(feature)) for feature in features]
        tl_heatmaps = [_sigmoid(self.subnet_tl_heats(feature)) for feature in features]
        br_heatmaps = [_sigmoid(self.subnet_br_heats(feature)) for feature in features]
        return tl_heatmaps, br_heatmaps, tl_corners_regr, br_corners_regr, tl_centers_regr, br_centers_regr


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=20):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    topk_clses = (topk_inds / (height * width)).int()
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _decode(tl_heats, br_heats, tl_regrs, br_regrs, tl_centers_regrs, br_centers_regrs, K=100, kernel=1, dist_threshold=0.2, num_dets=1000, layers_range=None, output_kernel_size=None, output_sizes=None, input_size=None, base_layer_range=None):
    top_k = K
    batch, cat, height_0, width_0 = tl_heats[0].size()
    for i in range(len(tl_heats)):
        tl_heat = tl_heats[i]
        br_heat = br_heats[i]
        tl_regr = tl_regrs[i]
        br_regr = br_regrs[i]
        tl_centers_regr = tl_centers_regrs[i]
        br_centers_regr = br_centers_regrs[i]
        batch, cat, height, width = tl_heat.size()
        height_scale = height_0 / height
        width_scale = width_0 / width
        tl_heat = _nms(tl_heat, kernel=output_kernel_size)
        br_heat = _nms(br_heat, kernel=output_kernel_size)
        tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
        br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
        tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
        tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
        br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
        br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)
        if tl_regr is not None and br_regr is not None and tl_centers_regr is not None and br_centers_regr is not None:
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            tl_regr = tl_regr.view(batch, K, 1, 2)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
            br_regr = br_regr.view(batch, 1, K, 2)
            tl_centers_regr = _tranpose_and_gather_feat(tl_centers_regr, tl_inds)
            tl_centers_regr = tl_centers_regr.view(batch, K, 1, 2)
            br_centers_regr = _tranpose_and_gather_feat(br_centers_regr, br_inds)
            br_centers_regr = br_centers_regr.view(batch, 1, K, 2)
            tl_xs = tl_xs + tl_regr[..., 0]
            tl_ys = tl_ys + tl_regr[..., 1]
            br_xs = br_xs + br_regr[..., 0]
            br_ys = br_ys + br_regr[..., 1]
        bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)
        boxes_widths = br_xs - tl_xs
        boxes_heights = br_ys - tl_ys
        distsx = torch.abs(1 - output_sizes[-1][1] * (tl_centers_regr[..., 0] + br_centers_regr[..., 0]) / boxes_widths)
        distsy = torch.abs(1 - output_sizes[-1][0] * (tl_centers_regr[..., 1] + br_centers_regr[..., 1]) / boxes_heights)
        dists = torch.abs(br_centers_regr - tl_centers_regr)
        dists = dists[..., 1] + dists[..., 0]
        tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
        br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
        scores = (tl_scores + br_scores) / 2
        tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
        br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
        cls_inds = tl_clses != br_clses
        if layers_range != None:
            layer_range = layers_range[i]
            diff_x = br_xs - tl_xs
            diff_y = br_ys - tl_ys
            wrange_ind = (diff_x < 0.8 * layer_range[2]) | (diff_x > 1.3 * layer_range[3])
            hrange_ind = (diff_y < 0.8 * layer_range[0]) | (diff_y > 1.3 * layer_range[1])
            scores[wrange_ind] = -1
            scores[hrange_ind] = -1
        dist_inds = (distsx > dist_threshold) | (distsy > dist_threshold) | (dists > 0.25)
        width_inds = br_xs < tl_xs
        height_inds = br_ys < tl_ys
        scores[cls_inds] = -1
        scores[dist_inds] = -1
        scores[width_inds] = -1
        scores[height_inds] = -1
        scores = scores.view(batch, -1)
        scores, inds = torch.topk(scores, min(num_dets, scores.shape[1]))
        scores = scores.unsqueeze(2)
        bboxes = bboxes.view(batch, -1, 4)
        bboxes = _gather_feat(bboxes, inds)
        clses = tl_clses.contiguous().view(batch, -1, 1)
        clses = _gather_feat(clses, inds).float()
        tl_scores = tl_scores.contiguous().view(batch, -1, 1)
        tl_scores = _gather_feat(tl_scores, inds).float()
        br_scores = br_scores.contiguous().view(batch, -1, 1)
        br_scores = _gather_feat(br_scores, inds).float()
        bboxes[:, :, (0)] *= width_scale
        bboxes[:, :, (1)] *= height_scale
        bboxes[:, :, (2)] *= width_scale
        bboxes[:, :, (3)] *= height_scale
        if i == 0:
            detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
        else:
            detections = torch.cat([detections, torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)], dim=1)
    top_scores, top_inds = torch.topk(detections[:, :, (4)], 5 * num_dets)
    detections = _gather_feat(detections, top_inds)
    return detections


class model(nn.Module):

    def __init__(self, db):
        super(model, self).__init__()
        classes = db.configs['categories']
        resnet = db.configs['backbone']
        layers = db.configs['layers_range']
        self.net = MatrixNetCorners(classes, resnet, layers)
        self._decode = _decode

    def _train(self, *xs):
        image = xs[0][0]
        tl_inds = xs[1]
        br_inds = xs[2]
        outs = self.net.forward(image)
        for ind in range(len(tl_inds)):
            outs[2][ind] = _tranpose_and_gather_feat(outs[2][ind], tl_inds[ind])
            outs[3][ind] = _tranpose_and_gather_feat(outs[3][ind], br_inds[ind])
            outs[4][ind] = _tranpose_and_gather_feat(outs[4][ind], tl_inds[ind])
            outs[5][ind] = _tranpose_and_gather_feat(outs[5][ind], br_inds[ind])
        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0][0]
        outs = self.net.forward(image)
        return self._decode(*outs, **kwargs)

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)


def _neg_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)
    neg_weights = torch.pow(1 - gt[neg_inds], 4)
    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights
        num_pos = pos_inds.float().sum()
        num_neg = neg_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss)
    return loss, num_pos


def _regr_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)
    regr = regr[mask]
    gt_regr = gt_regr[mask]
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
    regr_loss = regr_loss
    return regr_loss, num


class MatrixNetAnchorsLoss(nn.Module):

    def __init__(self, corner_regr_weight=1, center_regr_weight=0.1, focal_loss=_neg_loss):
        super(MatrixNetAnchorsLoss, self).__init__()
        self.corner_regr_weight = corner_regr_weight
        self.center_regr_weight = center_regr_weight
        self.focal_loss = focal_loss
        self.regr_loss = _regr_loss

    def forward(self, outs, targets):
        focal_loss = 0
        corner_regr_loss = 0
        anchors_heats = outs[0]
        anchors_tl_corners_regrs = outs[1]
        anchors_br_corners_regrs = outs[2]
        gt_anchors_heat = targets[0]
        gt_tl_corners_regr = targets[1]
        gt_br_corners_regr = targets[2]
        gt_mask = targets[3]
        numf = 0
        numr = 0
        for i in range(len(anchors_heats)):
            floss, num = self.focal_loss([anchors_heats[i]], gt_anchors_heat[i])
            focal_loss += floss
            numf += num
            rloss, num = self.regr_loss(anchors_br_corners_regrs[i], gt_br_corners_regr[i], gt_mask[i])
            numr += num
            corner_regr_loss += rloss
            rloss, num = self.regr_loss(anchors_tl_corners_regrs[i], gt_tl_corners_regr[i], gt_mask[i])
            numr += num
            corner_regr_loss += rloss
        if numr > 0:
            corner_regr_loss = corner_regr_loss / numr
        if numf > 0:
            focal_loss = focal_loss / numf
        loss = focal_loss + corner_regr_loss
        return loss.unsqueeze(0)


class MatrixNetCornerLoss(nn.Module):

    def __init__(self, corner_regr_weight=1, center_regr_weight=0.1, focal_loss=_neg_loss):
        super(MatrixNetCornerLoss, self).__init__()
        self.corner_regr_weight = corner_regr_weight
        self.center_regr_weight = center_regr_weight
        self.focal_loss = focal_loss
        self.regr_loss = _regr_loss

    def forward(self, outs, targets):
        focal_loss = 0
        corner_regr_loss = 0
        center_regr_loss = 0
        tl_heats = outs[0]
        br_heats = outs[1]
        tl_regrs = outs[2]
        br_regrs = outs[3]
        center_tl_regrs = outs[4]
        center_br_regrs = outs[5]
        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_mask = targets[2]
        gt_tl_regr = targets[3]
        gt_br_regr = targets[4]
        gt_center_tl_regr = targets[5]
        gt_center_br_regr = targets[5]
        numf = 0
        numr = 0
        for i in range(len(tl_heats)):
            floss, num = self.focal_loss([tl_heats[i]], gt_tl_heat[i])
            focal_loss += floss
            numf += num
            floss, num = self.focal_loss([br_heats[i]], gt_br_heat[i])
            focal_loss += floss
            rloss, num = self.regr_loss(tl_regrs[i], gt_tl_regr[i], gt_mask[i])
            numr += num
            corner_regr_loss += rloss
            rloss, num = self.regr_loss(br_regrs[i], gt_br_regr[i], gt_mask[i])
            corner_regr_loss += rloss
            rloss, num = self.regr_loss(center_tl_regrs[i], gt_center_tl_regr[i], gt_mask[i])
            center_regr_loss += rloss
            rloss, num = self.regr_loss(center_br_regrs[i], gt_center_br_regr[i], gt_mask[i])
            center_regr_loss += rloss
        corner_regr_loss = self.corner_regr_weight * corner_regr_loss
        center_regr_loss = self.center_regr_weight * center_regr_loss
        if numr > 0:
            corner_regr_loss = corner_regr_loss / numr
            center_regr_loss = center_regr_loss / numr
        if numf > 0:
            focal_loss = focal_loss / numf
        loss = focal_loss + corner_regr_loss + center_regr_loss
        return loss.unsqueeze(0)


def scatter(inputs, target_gpus, dim=0, chunk_sizes=None):
    """
    Slices variables into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not variables. Does not
    support Tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, Variable):
            return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
        assert not torch.is_tensor(obj), 'Tensors not supported in scatter.'
        if isinstance(obj, tuple):
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list):
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict):
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]
    return scatter_map(inputs)


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0, chunk_sizes=None):
    """Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim, chunk_sizes) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim, chunk_sizes) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class DataParallel(Module):
    """Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is the
    same size (so that each GPU processes the same number of samples).

    See also: :ref:`cuda-nn-dataparallel-instead`

    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel EXCEPT Tensors. All variables will be scattered on dim
    specified (default 0). Primitive types will be broadcasted, but all
    other types will be a shallow copy and can be corrupted if written to in
    the model's forward pass.

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])

    Example::

        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0, chunk_sizes=None):
        super(DataParallel, self).__init__()
        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.chunk_sizes = chunk_sizes
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids, self.chunk_sizes)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids, chunk_sizes):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim, chunk_sizes=self.chunk_sizes)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


class Network(nn.Module):

    def __init__(self, model, loss):
        super(Network, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, xs, ys, **kwargs):
        preds = self.model(*xs, **kwargs)
        loss = self.loss(preds, ys, **kwargs)
        return loss


class DummyModule(nn.Module):

    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model

    def forward(self, *xs, **kwargs):
        return self.module(*xs, **kwargs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DataParallel,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (DummyModule,
     lambda: ([], {'model': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
]

class Test_arashwan_matrixnet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

