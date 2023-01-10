import sys
_module = sys.modules[__name__]
del sys
dataset = _module
range_transform = _module
reseed = _module
static_dataset = _module
tps = _module
util = _module
vos_dataset = _module
eval = _module
inference = _module
data = _module
mask_mapper = _module
test_datasets = _module
video_reader = _module
inference_core = _module
interact = _module
fbrs = _module
controller = _module
clicker = _module
evaluation = _module
predictors = _module
base = _module
brs = _module
brs_functors = _module
brs_losses = _module
transforms = _module
base = _module
crops = _module
flip = _module
limit_longest_side = _module
zoom_in = _module
utils = _module
model = _module
initializer = _module
is_deeplab_model = _module
is_hrnet_model = _module
losses = _module
metrics = _module
modeling = _module
basic_blocks = _module
deeplab_v3 = _module
hrnet_ocr = _module
ocr = _module
resnet = _module
resnetv1b = _module
ops = _module
syncbn = _module
modules = _module
functional = _module
_csrc = _module
syncbn = _module
nn = _module
syncbn = _module
cython = _module
dist_maps = _module
misc = _module
vis = _module
fbrs_controller = _module
gui = _module
gui_utils = _module
interaction = _module
interactive_utils = _module
resource_manager = _module
s2m = _module
_deeplab = _module
s2m_network = _module
s2m_resnet = _module
utils = _module
s2m_controller = _module
timer = _module
kv_memory_store = _module
memory_manager = _module
interactive_demo = _module
merge_multi_scale = _module
aggregate = _module
cbam = _module
group_modules = _module
losses = _module
memory_util = _module
modules = _module
network = _module
resnet = _module
trainer = _module
scripts = _module
download_bl30k = _module
download_datasets = _module
expand_long_vid = _module
resize_youtube = _module
train = _module
configuration = _module
image_saver = _module
load_subset = _module
log_integrator = _module
logger = _module
palette = _module
tensor_util = _module

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


import random


from torch.utils.data.dataset import Dataset


from torchvision import transforms


from torchvision.transforms import InterpolationMode


import numpy as np


import torch.nn.functional as F


from torch.utils.data import DataLoader


from time import time


from scipy.optimize import fmin_l_bfgs_b


import math


import torch.nn as nn


from torch import nn


import torch._utils


from torch import nn as nn


import torch.cuda.comm as comm


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.nn import functional as F


from torch.nn.parameter import Parameter


from functools import partial


import functools


import time


from collections import OrderedDict


from typing import List


import warnings


from collections import defaultdict


from typing import Optional


from torch.utils import model_zoo


import torch.optim as optim


from torch.utils.data import ConcatDataset


import torch.distributed as distributed


import torchvision.transforms as transforms


from torch.utils.tensorboard import SummaryWriter


class BRSMaskLoss(torch.nn.Module):

    def __init__(self, eps=1e-05):
        super().__init__()
        self._eps = eps

    def forward(self, result, pos_mask, neg_mask):
        pos_diff = (1 - result) * pos_mask
        pos_target = torch.sum(pos_diff ** 2)
        pos_target = pos_target / (torch.sum(pos_mask) + self._eps)
        neg_diff = result * neg_mask
        neg_target = torch.sum(neg_diff ** 2)
        neg_target = neg_target / (torch.sum(neg_mask) + self._eps)
        loss = pos_target + neg_target
        with torch.no_grad():
            f_max_pos = torch.max(torch.abs(pos_diff)).item()
            f_max_neg = torch.max(torch.abs(neg_diff)).item()
        return loss, f_max_pos, f_max_neg


class SigmoidBinaryCrossEntropyLoss(nn.Module):

    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        label = torch.where(sample_weight, label, torch.zeros_like(label))
        if not self._from_sigmoid:
            loss = torch.relu(pred) - pred * label + F.softplus(-torch.abs(pred))
        else:
            eps = 1e-12
            loss = -(torch.log(pred + eps) * label + torch.log(1.0 - pred + eps) * (1.0 - label))
        loss = self._weight * (loss * sample_weight)
        return torch.mean(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))


class OracleMaskLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.gt_mask = None
        self.loss = SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
        self.predictor = None
        self.history = []

    def set_gt_mask(self, gt_mask):
        self.gt_mask = gt_mask
        self.history = []

    def forward(self, result, pos_mask, neg_mask):
        gt_mask = self.gt_mask
        if self.predictor.object_roi is not None:
            r1, r2, c1, c2 = self.predictor.object_roi[:4]
            gt_mask = gt_mask[:, :, r1:r2 + 1, c1:c2 + 1]
            gt_mask = torch.nn.functional.interpolate(gt_mask, result.size()[2:], mode='bilinear', align_corners=True)
        if result.shape[0] == 2:
            gt_mask_flipped = torch.flip(gt_mask, dims=[3])
            gt_mask = torch.cat([gt_mask, gt_mask_flipped], dim=0)
        loss = self.loss(result, gt_mask)
        self.history.append(loss.detach().cpu().numpy()[0])
        if len(self.history) > 5 and abs(self.history[-5] - self.history[-1]) < 1e-05:
            return 0, 0, 0
        return loss, 1.0, 1.0


class DistMaps(nn.Module):

    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode

    def get_coord_features(self, points, batchsize, rows, cols):
        if self.cpu_mode:
            coords = []
            for i in range(batchsize):
                norm_delimeter = self.spatial_scale * self.norm_radius
                coords.append(get_dist_maps(points[i].cpu().float().numpy(), rows, cols, norm_delimeter))
            coords = torch.from_numpy(np.stack(coords, axis=0)).float()
        else:
            num_points = points.shape[1] // 2
            points = points.view(-1, 2)
            invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0
            row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
            col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)
            coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
            coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1)
            add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)
            coords.add_(-add_xy)
            coords.div_(self.norm_radius * self.spatial_scale)
            coords.mul_(coords)
            coords[:, 0] += coords[:, 1]
            coords = coords[:, :1]
            coords[invalid_points, :, :, :] = 1000000.0
            coords = coords.view(-1, num_points, 1, rows, cols)
            coords = coords.min(dim=1)[0]
            coords = coords.view(-1, 2, rows, cols)
        coords.sqrt_().mul_(2).tanh_()
        return coords

    def forward(self, x, coords):
        return self.get_coord_features(coords, x.shape[0], x.shape[2], x.shape[3])


class DistMapsModel(nn.Module):

    def __init__(self, feature_extractor, head, norm_layer=nn.BatchNorm2d, use_rgb_conv=True, cpu_dist_maps=False, norm_radius=260):
        super(DistMapsModel, self).__init__()
        if use_rgb_conv:
            self.rgb_conv = nn.Sequential(nn.Conv2d(in_channels=5, out_channels=8, kernel_size=1), nn.LeakyReLU(negative_slope=0.2), norm_layer(8), nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1))
        else:
            self.rgb_conv = None
        self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0, cpu_mode=cpu_dist_maps)
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, image, points):
        coord_features = self.dist_maps(image, points)
        if self.rgb_conv is not None:
            x = self.rgb_conv(torch.cat((image, coord_features), dim=1))
        else:
            c1, c2 = torch.chunk(coord_features, 2, dim=1)
            c3 = torch.ones_like(c1)
            coord_features = torch.cat((c1, c2, c3), dim=1)
            x = 0.8 * image * coord_features + 0.2 * image
        backbone_features = self.feature_extractor(x)
        instance_out = self.head(backbone_features[0])
        instance_out = nn.functional.interpolate(instance_out, size=image.size()[2:], mode='bilinear', align_corners=True)
        return {'instances': instance_out}

    def load_weights(self, path_to_weights):
        current_state_dict = self.state_dict()
        new_state_dict = torch.load(path_to_weights, map_location='cpu')
        current_state_dict.update(new_state_dict)
        self.load_state_dict(current_state_dict)

    def get_trainable_params(self):
        backbone_params = nn.ParameterList()
        other_params = nn.ParameterList()
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)
        return backbone_params, other_params


class DistMapsHRNetModel(nn.Module):

    def __init__(self, feature_extractor, use_rgb_conv=True, with_aux_output=False, norm_layer=nn.BatchNorm2d, norm_radius=260, cpu_dist_maps=False):
        super(DistMapsHRNetModel, self).__init__()
        self.with_aux_output = with_aux_output
        if use_rgb_conv:
            self.rgb_conv = nn.Sequential(nn.Conv2d(in_channels=5, out_channels=8, kernel_size=1), nn.LeakyReLU(negative_slope=0.2), norm_layer(8), nn.Conv2d(in_channels=8, out_channels=3, kernel_size=1))
        else:
            self.rgb_conv = None
        self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0, cpu_mode=cpu_dist_maps)
        self.feature_extractor = feature_extractor

    def forward(self, image, points):
        coord_features = self.dist_maps(image, points)
        if self.rgb_conv is not None:
            x = self.rgb_conv(torch.cat((image, coord_features), dim=1))
        else:
            c1, c2 = torch.chunk(coord_features, 2, dim=1)
            c3 = torch.ones_like(c1)
            coord_features = torch.cat((c1, c2, c3), dim=1)
            x = 0.8 * image * coord_features + 0.2 * image
        feature_extractor_out = self.feature_extractor(x)
        instance_out = feature_extractor_out[0]
        instance_out = nn.functional.interpolate(instance_out, size=image.size()[2:], mode='bilinear', align_corners=True)
        outputs = {'instances': instance_out}
        if self.with_aux_output:
            instance_aux_out = feature_extractor_out[1]
            instance_aux_out = nn.functional.interpolate(instance_aux_out, size=image.size()[2:], mode='bilinear', align_corners=True)
            outputs['instances_aux'] = instance_aux_out
        return outputs

    def load_weights(self, path_to_weights):
        current_state_dict = self.state_dict()
        new_state_dict = torch.load(path_to_weights)
        current_state_dict.update(new_state_dict)
        self.load_state_dict(current_state_dict)

    def get_trainable_params(self):
        backbone_params = nn.ParameterList()
        other_params = nn.ParameterList()
        other_params_keys = []
        nonbackbone_keywords = ['rgb_conv', 'aux_head', 'cls_head', 'conv3x3_ocr', 'ocr_distri_head']
        for name, param in self.named_parameters():
            if param.requires_grad:
                if any(x in name for x in nonbackbone_keywords):
                    other_params.append(param)
                    other_params_keys.append(name)
                else:
                    backbone_params.append(param)
        None
        return backbone_params, other_params


class NormalizedFocalLossSigmoid(nn.Module):

    def __init__(self, axis=-1, alpha=0.25, gamma=2, from_logits=False, batch_axis=0, weight=None, size_average=True, detach_delimeter=True, eps=1e-12, scale=1.0, ignore_label=-1):
        super(NormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis
        self._scale = scale
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._k_sum = 0

    def forward(self, pred, label, sample_weight=None):
        one_hot = label > 0
        sample_weight = label != self._ignore_label
        if not self._from_logits:
            pred = torch.sigmoid(pred)
        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(one_hot, pred, 1 - pred)
        pt = torch.where(sample_weight, pt, torch.ones_like(pt))
        beta = (1 - pt) ** self._gamma
        sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        ignore_area = torch.sum(label == self._ignore_label, dim=tuple(range(1, label.dim()))).cpu().numpy()
        sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
        if np.any(ignore_area == 0):
            self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()
        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float)))
        loss = self._weight * (loss * sample_weight)
        if self._size_average:
            bsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))
        return self._scale * loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)


class FocalLoss(nn.Module):

    def __init__(self, axis=-1, alpha=0.25, gamma=2, from_logits=False, batch_axis=0, weight=None, num_class=None, eps=1e-09, size_average=True, scale=1.0):
        super(FocalLoss, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis
        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def forward(self, pred, label, sample_weight=None):
        if not self._from_logits:
            pred = F.sigmoid(pred)
        one_hot = label > 0
        pt = torch.where(one_hot, pred, 1 - pred)
        t = label != -1
        alpha = torch.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        beta = (1 - pt) ** self._gamma
        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float)))
        sample_weight = label != -1
        loss = self._weight * (loss * sample_weight)
        if self._size_average:
            tsum = torch.sum(label == 1, dim=misc.get_dims_with_exclusion(label.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (tsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))
        return self._scale * loss


class ConvHead(nn.Module):

    def __init__(self, out_channels, in_channels=32, num_layers=1, kernel_size=3, padding=1, norm_layer=nn.BatchNorm2d):
        super(ConvHead, self).__init__()
        convhead = []
        for i in range(num_layers):
            convhead.extend([nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding), nn.ReLU(), norm_layer(in_channels) if norm_layer is not None else nn.Identity()])
        convhead.append(nn.Conv2d(in_channels, out_channels, 1, padding=0))
        self.convhead = nn.Sequential(*convhead)

    def forward(self, *inputs):
        return self.convhead(inputs[0])


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_padding, dw_stride=1, activation=None, use_bias=False, norm_layer=None):
        super(SeparableConv2d, self).__init__()
        _activation = ops.select_activation_function(activation)
        self.body = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=dw_kernel, stride=dw_stride, padding=dw_padding, bias=use_bias, groups=in_channels), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=use_bias), norm_layer(out_channels) if norm_layer is not None else nn.Identity(), _activation())

    def forward(self, x):
        return self.body(x)


class SepConvHead(nn.Module):

    def __init__(self, num_outputs, in_channels, mid_channels, num_layers=1, kernel_size=3, padding=1, dropout_ratio=0.0, dropout_indx=0, norm_layer=nn.BatchNorm2d):
        super(SepConvHead, self).__init__()
        sepconvhead = []
        for i in range(num_layers):
            sepconvhead.append(SeparableConv2d(in_channels=in_channels if i == 0 else mid_channels, out_channels=mid_channels, dw_kernel=kernel_size, dw_padding=padding, norm_layer=norm_layer, activation='relu'))
            if dropout_ratio > 0 and dropout_indx == i:
                sepconvhead.append(nn.Dropout(dropout_ratio))
        sepconvhead.append(nn.Conv2d(in_channels=mid_channels, out_channels=num_outputs, kernel_size=1, padding=0))
        self.layers = nn.Sequential(*sepconvhead)

    def forward(self, *inputs):
        x = inputs[0]
        return self.layers(x)


class BottleneckV1b(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
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
        out = out + residual
        out = self.relu(out)
        return out


GLUON_RESNET_TORCH_HUB = 'rwightman/pytorch-pretrained-gluonresnet'


class ResNetV1b(nn.Module):
    """ Pre-trained ResNetV1b Model, which produces the strides of 8 featuremaps at conv5.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm2d`)
    deep_stem : bool, default False
        Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    final_drop : float, default 0.0
        Dropout ratio before the final classification layer.

    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    def __init__(self, block, layers, classes=1000, dilated=True, deep_stem=False, stem_width=32, avg_down=False, final_drop=0.0, norm_layer=nn.BatchNorm2d):
        self.inplanes = stem_width * 2 if deep_stem else 64
        super(ResNetV1b, self).__init__()
        if not deep_stem:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False), norm_layer(stem_width), nn.ReLU(True), nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False), norm_layer(stem_width), nn.ReLU(True), nn.Conv2d(stem_width, 2 * stem_width, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], avg_down=avg_down, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, avg_down=avg_down, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, avg_down=avg_down, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, avg_down=avg_down, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, avg_down=avg_down, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, avg_down=avg_down, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = None
        if final_drop > 0.0:
            self.drop = nn.Dropout(final_drop)
        self.fc = nn.Linear(512 * block.expansion, classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, avg_down=False, norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = []
            if avg_down:
                if dilation == 1:
                    downsample.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False))
                else:
                    downsample.append(nn.AvgPool2d(kernel_size=1, stride=1, ceil_mode=True, count_include_pad=False))
                downsample.extend([nn.Conv2d(self.inplanes, out_channels=planes * block.expansion, kernel_size=1, stride=1, bias=False), norm_layer(planes * block.expansion)])
                downsample = nn.Sequential(*downsample)
            else:
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, out_channels=planes * block.expansion, kernel_size=1, stride=stride, bias=False), norm_layer(planes * block.expansion))
        layers = []
        if dilation in (1, 2):
            layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2, downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc(x)
        return x


def _safe_state_dict_filtering(orig_dict, model_dict_keys):
    filtered_orig_dict = {}
    for k, v in orig_dict.items():
        if k in model_dict_keys:
            filtered_orig_dict[k] = v
        else:
            None
    return filtered_orig_dict


def resnet101_v1s(pretrained=False, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], deep_stem=True, stem_width=64, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        filtered_orig_dict = _safe_state_dict_filtering(torch.hub.load(GLUON_RESNET_TORCH_HUB, 'gluon_resnet101_v1s', pretrained=True).state_dict(), model_dict.keys())
        model_dict.update(filtered_orig_dict)
        model.load_state_dict(model_dict)
    return model


def resnet152_v1s(pretrained=False, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 8, 36, 3], deep_stem=True, stem_width=64, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        filtered_orig_dict = _safe_state_dict_filtering(torch.hub.load(GLUON_RESNET_TORCH_HUB, 'gluon_resnet152_v1s', pretrained=True).state_dict(), model_dict.keys())
        model_dict.update(filtered_orig_dict)
        model.load_state_dict(model_dict)
    return model


class BasicBlockV1b(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
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
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out


def resnet34_v1b(pretrained=False, **kwargs):
    model = ResNetV1b(BasicBlockV1b, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        filtered_orig_dict = _safe_state_dict_filtering(torch.hub.load(GLUON_RESNET_TORCH_HUB, 'gluon_resnet34_v1b', pretrained=True).state_dict(), model_dict.keys())
        model_dict.update(filtered_orig_dict)
        model.load_state_dict(model_dict)
    return model


def resnet50_v1s(pretrained=False, **kwargs):
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, stem_width=64, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        filtered_orig_dict = _safe_state_dict_filtering(torch.hub.load(GLUON_RESNET_TORCH_HUB, 'gluon_resnet50_v1s', pretrained=True).state_dict(), model_dict.keys())
        model_dict.update(filtered_orig_dict)
        model.load_state_dict(model_dict)
    return model


class ResNetBackbone(torch.nn.Module):

    def __init__(self, backbone='resnet50', pretrained_base=True, dilated=True, **kwargs):
        super(ResNetBackbone, self).__init__()
        if backbone == 'resnet34':
            pretrained = resnet34_v1b(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet50':
            pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet101':
            pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        elif backbone == 'resnet152':
            pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
        else:
            raise RuntimeError(f'unknown backbone: {backbone}')
        self.conv1 = pretrained.conv1
        self.bn1 = pretrained.bn1
        self.relu = pretrained.relu
        self.maxpool = pretrained.maxpool
        self.layer1 = pretrained.layer1
        self.layer2 = pretrained.layer2
        self.layer3 = pretrained.layer3
        self.layer4 = pretrained.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4


def _ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=atrous_rate, dilation=atrous_rate, bias=False), norm_layer(out_channels), nn.ReLU())
    return block


class _AsppPooling(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False), norm_layer(out_channels), nn.ReLU())

    def forward(self, x):
        pool = self.gap(x)
        return F.interpolate(pool, x.size()[2:], mode='bilinear', align_corners=True)


class _ASPP(nn.Module):

    def __init__(self, in_channels, atrous_rates, out_channels=256, project_dropout=0.5, norm_layer=nn.BatchNorm2d):
        super(_ASPP, self).__init__()
        b0 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False), norm_layer(out_channels), nn.ReLU())
        rate1, rate2, rate3 = tuple(atrous_rates)
        b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer)
        b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer)
        b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer)
        b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer)
        self.concurent = nn.ModuleList([b0, b1, b2, b3, b4])
        project = [nn.Conv2d(in_channels=5 * out_channels, out_channels=out_channels, kernel_size=1, bias=False), norm_layer(out_channels), nn.ReLU()]
        if project_dropout > 0:
            project.append(nn.Dropout(project_dropout))
        self.project = nn.Sequential(*project)

    def forward(self, x):
        x = torch.cat([block(x) for block in self.concurent], dim=1)
        return self.project(x)


class _DeepLabHead(nn.Module):

    def __init__(self, out_channels, in_channels, mid_channels=256, norm_layer=nn.BatchNorm2d):
        super(_DeepLabHead, self).__init__()
        self.block = nn.Sequential(SeparableConv2d(in_channels=in_channels, out_channels=mid_channels, dw_kernel=3, dw_padding=1, activation='relu', norm_layer=norm_layer), SeparableConv2d(in_channels=mid_channels, out_channels=mid_channels, dw_kernel=3, dw_padding=1, activation='relu', norm_layer=norm_layer), nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1))

    def forward(self, x):
        return self.block(x)


class _SkipProject(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(_SkipProject, self).__init__()
        _activation = ops.select_activation_function('relu')
        self.skip_project = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), norm_layer(out_channels), _activation())

    def forward(self, x):
        return self.skip_project(x)


class DeepLabV3Plus(nn.Module):

    def __init__(self, backbone='resnet50', norm_layer=nn.BatchNorm2d, backbone_norm_layer=None, ch=256, project_dropout=0.5, inference_mode=False, **kwargs):
        super(DeepLabV3Plus, self).__init__()
        if backbone_norm_layer is None:
            backbone_norm_layer = norm_layer
        self.backbone_name = backbone
        self.norm_layer = norm_layer
        self.backbone_norm_layer = backbone_norm_layer
        self.inference_mode = False
        self.ch = ch
        self.aspp_in_channels = 2048
        self.skip_project_in_channels = 256
        self._kwargs = kwargs
        if backbone == 'resnet34':
            self.aspp_in_channels = 512
            self.skip_project_in_channels = 64
        self.backbone = ResNetBackbone(backbone=self.backbone_name, pretrained_base=False, norm_layer=self.backbone_norm_layer, **kwargs)
        self.head = _DeepLabHead(in_channels=ch + 32, mid_channels=ch, out_channels=ch, norm_layer=self.norm_layer)
        self.skip_project = _SkipProject(self.skip_project_in_channels, 32, norm_layer=self.norm_layer)
        self.aspp = _ASPP(in_channels=self.aspp_in_channels, atrous_rates=[12, 24, 36], out_channels=ch, project_dropout=project_dropout, norm_layer=self.norm_layer)
        if inference_mode:
            self.set_prediction_mode()

    def load_pretrained_weights(self):
        pretrained = ResNetBackbone(backbone=self.backbone_name, pretrained_base=True, norm_layer=self.backbone_norm_layer, **self._kwargs)
        backbone_state_dict = self.backbone.state_dict()
        pretrained_state_dict = pretrained.state_dict()
        backbone_state_dict.update(pretrained_state_dict)
        self.backbone.load_state_dict(backbone_state_dict)
        if self.inference_mode:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def set_prediction_mode(self):
        self.inference_mode = True
        self.eval()

    def forward(self, x):
        with ExitStack() as stack:
            if self.inference_mode:
                stack.enter_context(torch.no_grad())
            c1, _, c3, c4 = self.backbone(x)
            c1 = self.skip_project(c1)
            x = self.aspp(c4)
            x = F.interpolate(x, c1.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, c1), dim=1)
            x = self.head(x)
        return x,


relu_inplace = True


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True, norm_layer=nn.BatchNorm2d, align_corners=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.norm_layer = norm_layer
        self.align_corners = align_corners
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=relu_inplace)

    def _check_branches(self, num_branches, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), self.norm_layer(num_channels[branch_index] * block.expansion))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample=downsample, norm_layer=self.norm_layer))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], norm_layer=self.norm_layer))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(nn.Conv2d(in_channels=num_inchannels[j], out_channels=num_inchannels[i], kernel_size=1, bias=False), self.norm_layer(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, kernel_size=3, stride=2, padding=1, bias=False), self.norm_layer(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, kernel_size=3, stride=2, padding=1, bias=False), self.norm_layer(num_outchannels_conv3x3), nn.ReLU(inplace=relu_inplace)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=[height_output, width_output], mode='bilinear', align_corners=self.align_corners)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)
        probs = F.softmax(self.scale * probs, dim=2)
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)
        return ocr_context


class ObjectAttentionBlock2D(nn.Module):
    """
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    """

    def __init__(self, in_channels, key_channels, scale=1, norm_layer=nn.BatchNorm2d, align_corners=True):
        super(ObjectAttentionBlock2D, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.align_corners = align_corners
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)), nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)))
        self.f_object = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)), nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)))
        self.f_down = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sequential(norm_layer(self.key_channels), nn.ReLU(inplace=True)))
        self.f_up = nn.Sequential(nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.Sequential(norm_layer(self.in_channels), nn.ReLU(inplace=True)))

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)
        sim_map = torch.matmul(query, key)
        sim_map = self.key_channels ** -0.5 * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=self.align_corners)
        return context


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.1, norm_layer=nn.BatchNorm2d, align_corners=True):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels, scale, norm_layer, align_corners)
        _in_channels = 2 * in_channels
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False), nn.Sequential(norm_layer(out_channels), nn.ReLU(inplace=True)), nn.Dropout2d(dropout))

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class HighResolutionNet(nn.Module):

    def __init__(self, width, num_classes, ocr_width=256, small=False, norm_layer=nn.BatchNorm2d, align_corners=True):
        super(HighResolutionNet, self).__init__()
        self.norm_layer = norm_layer
        self.width = width
        self.ocr_width = ocr_width
        self.align_corners = align_corners
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = norm_layer(64)
        self.relu = nn.ReLU(inplace=relu_inplace)
        num_blocks = 2 if small else 4
        stage1_num_channels = 64
        self.layer1 = self._make_layer(BottleneckV1b, 64, stage1_num_channels, blocks=num_blocks)
        stage1_out_channel = BottleneckV1b.expansion * stage1_num_channels
        self.stage2_num_branches = 2
        num_channels = [width, 2 * width]
        num_inchannels = [(num_channels[i] * BasicBlockV1b.expansion) for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_inchannels)
        self.stage2, pre_stage_channels = self._make_stage(BasicBlockV1b, num_inchannels=num_inchannels, num_modules=1, num_branches=self.stage2_num_branches, num_blocks=2 * [num_blocks], num_channels=num_channels)
        self.stage3_num_branches = 3
        num_channels = [width, 2 * width, 4 * width]
        num_inchannels = [(num_channels[i] * BasicBlockV1b.expansion) for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_inchannels)
        self.stage3, pre_stage_channels = self._make_stage(BasicBlockV1b, num_inchannels=num_inchannels, num_modules=3 if small else 4, num_branches=self.stage3_num_branches, num_blocks=3 * [num_blocks], num_channels=num_channels)
        self.stage4_num_branches = 4
        num_channels = [width, 2 * width, 4 * width, 8 * width]
        num_inchannels = [(num_channels[i] * BasicBlockV1b.expansion) for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_inchannels)
        self.stage4, pre_stage_channels = self._make_stage(BasicBlockV1b, num_inchannels=num_inchannels, num_modules=2 if small else 3, num_branches=self.stage4_num_branches, num_blocks=4 * [num_blocks], num_channels=num_channels)
        last_inp_channels = np.int(np.sum(pre_stage_channels))
        ocr_mid_channels = 2 * ocr_width
        ocr_key_channels = ocr_width
        self.conv3x3_ocr = nn.Sequential(nn.Conv2d(last_inp_channels, ocr_mid_channels, kernel_size=3, stride=1, padding=1), norm_layer(ocr_mid_channels), nn.ReLU(inplace=relu_inplace))
        self.ocr_gather_head = SpatialGather_Module(num_classes)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels, key_channels=ocr_key_channels, out_channels=ocr_mid_channels, scale=1, dropout=0.05, norm_layer=norm_layer, align_corners=align_corners)
        self.cls_head = nn.Conv2d(ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_head = nn.Sequential(nn.Conv2d(last_inp_channels, last_inp_channels, kernel_size=1, stride=1, padding=0), norm_layer(last_inp_channels), nn.ReLU(inplace=relu_inplace), nn.Conv2d(last_inp_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], kernel_size=3, stride=1, padding=1, bias=False), self.norm_layer(num_channels_cur_layer[i]), nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False), self.norm_layer(outchannels), nn.ReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), self.norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample, norm_layer=self.norm_layer))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=self.norm_layer))
        return nn.Sequential(*layers)

    def _make_stage(self, block, num_inchannels, num_modules, num_branches, num_blocks, num_channels, fuse_method='SUM', multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output, norm_layer=self.norm_layer, align_corners=self.align_corners))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        feats = self.compute_hrnet_feats(x)
        out_aux = self.aux_head(feats)
        feats = self.conv3x3_ocr(feats)
        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)
        out = self.cls_head(feats)
        return [out, out_aux]

    def compute_hrnet_feats(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_num_branches):
            if self.transition2[i] is not None:
                if i < self.stage2_num_branches:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_num_branches):
            if self.transition3[i] is not None:
                if i < self.stage3_num_branches:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=self.align_corners)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=self.align_corners)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=self.align_corners)
        return torch.cat([x[0], x1, x2, x3], 1)

    def load_pretrained_weights(self, pretrained_path=''):
        model_dict = self.state_dict()
        if not os.path.exists(pretrained_path):
            None
            None
            exit(1)
        pretrained_dict = torch.load(pretrained_path, map_location={'cuda:0': 'cpu'})
        pretrained_dict = {k.replace('last_layer', 'aux_head').replace('model.', ''): v for k, v in pretrained_dict.items()}
        None
        None
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


class BilinearConvTranspose2d(nn.ConvTranspose2d):

    def __init__(self, in_channels, out_channels, scale, groups=1):
        kernel_size = 2 * scale - scale % 2
        self.scale = scale
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=scale, padding=1, groups=groups, bias=False)
        self.apply(initializer.Bilinear(scale=scale, in_channels=in_channels, groups=groups))


class _BatchNorm(nn.Module):
    """
    Customized BatchNorm from nn.BatchNorm
    >> added freeze attribute to enable bn freeze.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.freezed = False
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        return NotImplemented

    def forward(self, input):
        self._check_input_dim(input)
        compute_stats = not self.freezed and self.training and self.track_running_stats
        ret = F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, compute_stats, self.momentum, self.eps)
        return ret

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}'.format(**self.__dict__)


class BatchNorm2dNoSync(_BatchNorm):
    """
    Equivalent to nn.BatchNorm2d
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


def _load_C_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir = os.path.join(this_dir, 'csrc')
    main_file = glob.glob(os.path.join(this_dir, '*.cpp'))
    sources_cpu = glob.glob(os.path.join(this_dir, 'cpu', '*.cpp'))
    sources_cuda = glob.glob(os.path.join(this_dir, 'cuda', '*.cu'))
    sources = main_file + sources_cpu
    extra_cflags = []
    extra_cuda_cflags = []
    if torch.cuda.is_available() and CUDA_HOME is not None:
        sources.extend(sources_cuda)
        extra_cflags = ['-O3', '-DWITH_CUDA']
        extra_cuda_cflags = ['--expt-extended-lambda']
    sources = [os.path.join(this_dir, s) for s in sources]
    extra_include_paths = [this_dir]
    return load(name='ext_lib', sources=sources, extra_cflags=extra_cflags, extra_include_paths=extra_include_paths, extra_cuda_cflags=extra_cuda_cflags)


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count


class BatchNorm2dSyncFunc(Function):

    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, extra, compute_stats=True, momentum=0.1, eps=1e-05):

        def _parse_extra(ctx, extra):
            ctx.is_master = extra['is_master']
            if ctx.is_master:
                ctx.master_queue = extra['master_queue']
                ctx.worker_queues = extra['worker_queues']
                ctx.worker_ids = extra['worker_ids']
            else:
                ctx.master_queue = extra['master_queue']
                ctx.worker_queue = extra['worker_queue']
        if extra is not None:
            _parse_extra(ctx, extra)
        ctx.compute_stats = compute_stats
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.affine = weight is not None and bias is not None
        if ctx.compute_stats:
            N = _count_samples(x) * (ctx.master_queue.maxsize + 1)
            assert N > 1
            xsum, xsqsum = _backend.syncbn_sum_sqsum(x.detach())
            if ctx.is_master:
                xsums, xsqsums = [xsum], [xsqsum]
                for _ in range(ctx.master_queue.maxsize):
                    xsum_w, xsqsum_w = ctx.master_queue.get()
                    ctx.master_queue.task_done()
                    xsums.append(xsum_w)
                    xsqsums.append(xsqsum_w)
                xsum = comm.reduce_add(xsums)
                xsqsum = comm.reduce_add(xsqsums)
                mean = xsum / N
                sumvar = xsqsum - xsum * mean
                var = sumvar / N
                uvar = sumvar / (N - 1)
                tensors = comm.broadcast_coalesced((mean, uvar, var), [mean.get_device()] + ctx.worker_ids)
                for ts, queue in zip(tensors[1:], ctx.worker_queues):
                    queue.put(ts)
            else:
                ctx.master_queue.put((xsum, xsqsum))
                mean, uvar, var = ctx.worker_queue.get()
                ctx.worker_queue.task_done()
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * mean)
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * uvar)
            ctx.N = N
            ctx.save_for_backward(x, weight, bias, mean, var)
        else:
            mean, var = running_mean, running_var
        z = _backend.syncbn_forward(x, weight, bias, mean, var, ctx.affine, ctx.eps)
        return z

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        x, weight, bias, mean, var = ctx.saved_tensors
        dz = dz.contiguous()
        sum_dz, sum_dz_xhat = _backend.syncbn_backward_xhat(dz, x, mean, var, ctx.eps)
        if ctx.is_master:
            sum_dzs, sum_dz_xhats = [sum_dz], [sum_dz_xhat]
            for _ in range(ctx.master_queue.maxsize):
                sum_dz_w, sum_dz_xhat_w = ctx.master_queue.get()
                ctx.master_queue.task_done()
                sum_dzs.append(sum_dz_w)
                sum_dz_xhats.append(sum_dz_xhat_w)
            sum_dz = comm.reduce_add(sum_dzs)
            sum_dz_xhat = comm.reduce_add(sum_dz_xhats)
            sum_dz /= ctx.N
            sum_dz_xhat /= ctx.N
            tensors = comm.broadcast_coalesced((sum_dz, sum_dz_xhat), [mean.get_device()] + ctx.worker_ids)
            for ts, queue in zip(tensors[1:], ctx.worker_queues):
                queue.put(ts)
        else:
            ctx.master_queue.put((sum_dz, sum_dz_xhat))
            sum_dz, sum_dz_xhat = ctx.worker_queue.get()
            ctx.worker_queue.task_done()
        dx, dweight, dbias = _backend.syncbn_backward(dz, x, weight, bias, mean, var, sum_dz, sum_dz_xhat, ctx.affine, ctx.eps)
        return dx, dweight, dbias, None, None, None, None, None, None


batchnorm2d_sync = BatchNorm2dSyncFunc.apply


class BatchNorm2dSync(BatchNorm2dNoSync):
    """
    BatchNorm2d with automatic multi-GPU Sync
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2dSync, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.sync_enabled = True
        self.devices = list(range(torch.cuda.device_count()))
        if len(self.devices) > 1:
            self.worker_ids = self.devices[1:]
            self.master_queue = Queue(len(self.worker_ids))
            self.worker_queues = [Queue(1) for _ in self.worker_ids]

    def forward(self, x):
        compute_stats = not self.freezed and self.training and self.track_running_stats
        if self.sync_enabled and compute_stats and len(self.devices) > 1:
            if x.get_device() == self.devices[0]:
                extra = {'is_master': True, 'master_queue': self.master_queue, 'worker_queues': self.worker_queues, 'worker_ids': self.worker_ids}
            else:
                extra = {'is_master': False, 'master_queue': self.master_queue, 'worker_queue': self.worker_queues[self.worker_ids.index(x.get_device())]}
            return batchnorm2d_sync(x, self.weight, self.bias, self.running_mean, self.running_var, extra, compute_stats, self.momentum, self.eps)
        return super(BatchNorm2dSync, self).forward(x)

    def __repr__(self):
        """repr"""
        rep = '{name}({num_features}, eps={eps}, momentum={momentum},affine={affine}, track_running_stats={track_running_stats},devices={devices})'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


class ASPPConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation):
        modules = [nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):

    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(0.1))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabHeadV3Plus(nn.Module):

    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.aspp = ASPP(in_channels, aspp_dilate)
        self.classifier = nn.Sequential(nn.Conv2d(304, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, num_classes, 1))
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHead(nn.Module):

    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()
        self.classifier = nn.Sequential(ASPP(in_channels, aspp_dilate), nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, num_classes, 1))
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature['out'])

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels), nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias))
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
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


class ResNet(nn.Module):

    def __init__(self, block, layers=(3, 4, 23, 3), extra_dim=0):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3 + extra_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
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

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)


class _SimpleSegmentationModel(nn.Module):

    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError('return_layers are not present in model')
        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(), nn.Linear(gate_channels, gate_channels // reduction_ratio), nn.ReLU(), nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class GConv2D(nn.Conv2d):

    def forward(self, g):
        batch_size, num_objects = g.shape[:2]
        g = super().forward(g.flatten(start_dim=0, end_dim=1))
        return g.view(batch_size, num_objects, *g.shape[1:])


class GroupResBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        if in_dim == out_dim:
            self.downsample = None
        else:
            self.downsample = GConv2D(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv1 = GConv2D(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = GConv2D(out_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, g):
        out_g = self.conv1(F.relu(g))
        out_g = self.conv2(F.relu(out_g))
        if self.downsample is not None:
            g = self.downsample(g)
        return out_g + g


class MainToGroupDistributor(nn.Module):

    def __init__(self, x_transform=None, method='cat', reverse_order=False):
        super().__init__()
        self.x_transform = x_transform
        self.method = method
        self.reverse_order = reverse_order

    def forward(self, x, g):
        num_objects = g.shape[1]
        if self.x_transform is not None:
            x = self.x_transform(x)
        if self.method == 'cat':
            if self.reverse_order:
                g = torch.cat([g, x.unsqueeze(1).expand(-1, num_objects, -1, -1, -1)], 2)
            else:
                g = torch.cat([x.unsqueeze(1).expand(-1, num_objects, -1, -1, -1), g], 2)
        elif self.method == 'add':
            g = x.unsqueeze(1).expand(-1, num_objects, -1, -1, -1) + g
        else:
            raise NotImplementedError
        return g


class BootstrappedCE(nn.Module):

    def __init__(self, start_warm, end_warm, top_p=0.15):
        super().__init__()
        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0
        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()
        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1 - self.top_p) * ((self.end_warm - it) / (self.end_warm - self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class FeatureFusionBlock(nn.Module):

    def __init__(self, x_in_dim, g_in_dim, g_mid_dim, g_out_dim):
        super().__init__()
        self.distributor = MainToGroupDistributor()
        self.block1 = GroupResBlock(x_in_dim + g_in_dim, g_mid_dim)
        self.attention = CBAM(g_mid_dim)
        self.block2 = GroupResBlock(g_mid_dim, g_out_dim)

    def forward(self, x, g):
        batch_size, num_objects = g.shape[:2]
        g = self.distributor(x, g)
        g = self.block1(g)
        r = self.attention(g.flatten(start_dim=0, end_dim=1))
        r = r.view(batch_size, num_objects, *r.shape[1:])
        g = self.block2(g + r)
        return g


def interpolate_groups(g, ratio, mode, align_corners):
    batch_size, num_objects = g.shape[:2]
    g = F.interpolate(g.flatten(start_dim=0, end_dim=1), scale_factor=ratio, mode=mode, align_corners=align_corners)
    g = g.view(batch_size, num_objects, *g.shape[1:])
    return g


def downsample_groups(g, ratio=1 / 2, mode='area', align_corners=None):
    return interpolate_groups(g, ratio, mode, align_corners)


class HiddenUpdater(nn.Module):

    def __init__(self, g_dims, mid_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.g16_conv = GConv2D(g_dims[0], mid_dim, kernel_size=1)
        self.g8_conv = GConv2D(g_dims[1], mid_dim, kernel_size=1)
        self.g4_conv = GConv2D(g_dims[2], mid_dim, kernel_size=1)
        self.transform = GConv2D(mid_dim + hidden_dim, hidden_dim * 3, kernel_size=3, padding=1)
        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = self.g16_conv(g[0]) + self.g8_conv(downsample_groups(g[1], ratio=1 / 2)) + self.g4_conv(downsample_groups(g[2], ratio=1 / 4))
        g = torch.cat([g, h], 2)
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:, :, :self.hidden_dim])
        update_gate = torch.sigmoid(values[:, :, self.hidden_dim:self.hidden_dim * 2])
        new_value = torch.tanh(values[:, :, self.hidden_dim * 2:])
        new_h = forget_gate * h * (1 - update_gate) + update_gate * new_value
        return new_h


class HiddenReinforcer(nn.Module):

    def __init__(self, g_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transform = GConv2D(g_dim + hidden_dim, hidden_dim * 3, kernel_size=3, padding=1)
        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = torch.cat([g, h], 2)
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:, :, :self.hidden_dim])
        update_gate = torch.sigmoid(values[:, :, self.hidden_dim:self.hidden_dim * 2])
        new_value = torch.tanh(values[:, :, self.hidden_dim * 2:])
        new_h = forget_gate * h * (1 - update_gate) + update_gate * new_value
        return new_h


class ValueEncoder(nn.Module):

    def __init__(self, value_dim, hidden_dim, single_object=False):
        super().__init__()
        self.single_object = single_object
        network = resnet.resnet18(pretrained=True, extra_dim=1 if single_object else 2)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.distributor = MainToGroupDistributor()
        self.fuser = FeatureFusionBlock(1024, 256, value_dim, value_dim)
        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None

    def forward(self, image, image_feat_f16, h, masks, others, is_deep_update=True):
        if not self.single_object:
            g = torch.stack([masks, others], 2)
        else:
            g = masks.unsqueeze(2)
        g = self.distributor(image, g)
        batch_size, num_objects = g.shape[:2]
        g = g.flatten(start_dim=0, end_dim=1)
        g = self.conv1(g)
        g = self.bn1(g)
        g = self.maxpool(g)
        g = self.relu(g)
        g = self.layer1(g)
        g = self.layer2(g)
        g = self.layer3(g)
        g = g.view(batch_size, num_objects, *g.shape[1:])
        g = self.fuser(image_feat_f16, g)
        if is_deep_update and self.hidden_reinforce is not None:
            h = self.hidden_reinforce(g, h)
        return g, h


class KeyEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        network = resnet.resnet50(pretrained=True)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu
        self.maxpool = network.maxpool
        self.res2 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3

    def forward(self, f):
        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f4 = self.res2(x)
        f8 = self.layer2(f4)
        f16 = self.layer3(f8)
        return f16, f8, f4


def upsample_groups(g, ratio=2, mode='bilinear', align_corners=False):
    return interpolate_groups(g, ratio, mode, align_corners)


class UpsampleBlock(nn.Module):

    def __init__(self, skip_dim, g_up_dim, g_out_dim, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_dim, g_up_dim, kernel_size=3, padding=1)
        self.distributor = MainToGroupDistributor(method='add')
        self.out_conv = GroupResBlock(g_up_dim, g_out_dim)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_g):
        skip_f = self.skip_conv(skip_f)
        g = upsample_groups(up_g, ratio=self.scale_factor)
        g = self.distributor(skip_f, g)
        g = self.out_conv(g)
        return g


class KeyProjection(nn.Module):

    def __init__(self, in_dim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)
        self.d_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        self.e_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)
        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x, need_s, need_e):
        shrinkage = self.d_proj(x) ** 2 + 1 if need_s else None
        selection = torch.sigmoid(self.e_proj(x)) if need_e else None
        return self.key_proj(x), shrinkage, selection


class Decoder(nn.Module):

    def __init__(self, val_dim, hidden_dim):
        super().__init__()
        self.fuser = FeatureFusionBlock(1024, val_dim + hidden_dim, 512, 512)
        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater([512, 256, 256 + 1], 256, hidden_dim)
        else:
            self.hidden_update = None
        self.up_16_8 = UpsampleBlock(512, 512, 256)
        self.up_8_4 = UpsampleBlock(256, 256, 256)
        self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, f16, f8, f4, hidden_state, memory_readout, h_out=True):
        batch_size, num_objects = memory_readout.shape[:2]
        if self.hidden_update is not None:
            g16 = self.fuser(f16, torch.cat([memory_readout, hidden_state], 2))
        else:
            g16 = self.fuser(f16, memory_readout)
        g8 = self.up_16_8(f8, g16)
        g4 = self.up_8_4(f4, g8)
        logits = self.pred(F.relu(g4.flatten(start_dim=0, end_dim=1)))
        if h_out and self.hidden_update is not None:
            g4 = torch.cat([g4, logits.view(batch_size, num_objects, 1, *logits.shape[-2:])], 2)
            hidden_state = self.hidden_update([g16, g8, g4], hidden_state)
        else:
            hidden_state = None
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])
        return hidden_state, logits


def aggregate(prob, dim, return_logits=False):
    new_prob = torch.cat([torch.prod(1 - prob, dim=dim, keepdim=True), prob], dim).clamp(1e-07, 1 - 1e-07)
    logits = torch.log(new_prob / (1 - new_prob))
    prob = F.softmax(logits, dim=dim)
    if return_logits:
        return logits, prob
    else:
        return prob


def do_softmax(similarity, top_k: Optional[int]=None, inplace=False, return_usage=False):
    if top_k is not None:
        values, indices = torch.topk(similarity, k=top_k, dim=1)
        x_exp = values.exp_()
        x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
        if inplace:
            similarity.zero_().scatter_(1, indices, x_exp)
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp)
    else:
        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(similarity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum
        indices = None
    if return_usage:
        return affinity, affinity.sum(dim=2)
    return affinity


def get_similarity(mk, ms, qk, qe):
    CK = mk.shape[1]
    mk = mk.flatten(start_dim=2)
    ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
    qk = qk.flatten(start_dim=2)
    qe = qe.flatten(start_dim=2) if qe is not None else None
    if qe is not None:
        mk = mk.transpose(1, 2)
        a_sq = mk.pow(2) @ qe
        two_ab = 2 * (mk @ (qk * qe))
        b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
        similarity = -a_sq + two_ab - b_sq
    else:
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        two_ab = 2 * (mk.transpose(1, 2) @ qk)
        similarity = -a_sq + two_ab
    if ms is not None:
        similarity = similarity * ms / math.sqrt(CK)
    else:
        similarity = similarity / math.sqrt(CK)
    return similarity


def get_affinity(mk, ms, qk, qe):
    similarity = get_similarity(mk, ms, qk, qe)
    affinity = do_softmax(similarity)
    return affinity


def readout(affinity, mv):
    B, CV, T, H, W = mv.shape
    mo = mv.view(B, CV, T * H * W)
    mem = torch.bmm(mo, affinity)
    mem = mem.view(B, CV, H, W)
    return mem


class XMem(nn.Module):

    def __init__(self, config, model_path=None, map_location=None):
        """
        model_path/map_location are used in evaluation only
        map_location is for converting models saved in cuda to cpu
        """
        super().__init__()
        model_weights = self.init_hyperparameters(config, model_path, map_location)
        self.single_object = config.get('single_object', False)
        None
        self.key_encoder = KeyEncoder()
        self.value_encoder = ValueEncoder(self.value_dim, self.hidden_dim, self.single_object)
        self.key_proj = KeyProjection(1024, self.key_dim)
        self.decoder = Decoder(self.value_dim, self.hidden_dim)
        if model_weights is not None:
            self.load_weights(model_weights, init_as_zero_if_needed=True)

    def encode_key(self, frame, need_sk=True, need_ek=True):
        if len(frame.shape) == 5:
            need_reshape = True
            b, t = frame.shape[:2]
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            need_reshape = False
        else:
            raise NotImplementedError
        f16, f8, f4 = self.key_encoder(frame)
        key, shrinkage, selection = self.key_proj(f16, need_sk, need_ek)
        if need_reshape:
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()
            if shrinkage is not None:
                shrinkage = shrinkage.view(b, t, *shrinkage.shape[-3:]).transpose(1, 2).contiguous()
            if selection is not None:
                selection = selection.view(b, t, *selection.shape[-3:]).transpose(1, 2).contiguous()
            f16 = f16.view(b, t, *f16.shape[-3:])
            f8 = f8.view(b, t, *f8.shape[-3:])
            f4 = f4.view(b, t, *f4.shape[-3:])
        return key, shrinkage, selection, f16, f8, f4

    def encode_value(self, frame, image_feat_f16, h16, masks, is_deep_update=True):
        num_objects = masks.shape[1]
        if num_objects != 1:
            others = torch.cat([torch.sum(masks[:, [j for j in range(num_objects) if i != j]], dim=1, keepdim=True) for i in range(num_objects)], 1)
        else:
            others = torch.zeros_like(masks)
        g16, h16 = self.value_encoder(frame, image_feat_f16, h16, masks, others, is_deep_update)
        return g16, h16

    def read_memory(self, query_key, query_selection, memory_key, memory_shrinkage, memory_value):
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        memory_value    : B * num_objects * CV * T * H * W
        """
        batch_size, num_objects = memory_value.shape[:2]
        memory_value = memory_value.flatten(start_dim=1, end_dim=2)
        affinity = get_affinity(memory_key, memory_shrinkage, query_key, query_selection)
        memory = readout(affinity, memory_value)
        memory = memory.view(batch_size, num_objects, self.value_dim, *memory.shape[-2:])
        return memory

    def segment(self, multi_scale_features, memory_readout, hidden_state, selector=None, h_out=True, strip_bg=True):
        hidden_state, logits = self.decoder(*multi_scale_features, hidden_state, memory_readout, h_out=h_out)
        prob = torch.sigmoid(logits)
        if selector is not None:
            prob = prob * selector
        logits, prob = aggregate(prob, dim=1, return_logits=True)
        if strip_bg:
            prob = prob[:, 1:]
        return hidden_state, logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'read_memory':
            return self.read_memory(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError

    def init_hyperparameters(self, config, model_path=None, map_location=None):
        """
        Init three hyperparameters: key_dim, value_dim, and hidden_dim
        If model_path is provided, we load these from the model weights
        The actual parameters are then updated to the config in-place

        Otherwise we load it either from the config or default
        """
        if model_path is not None:
            model_weights = torch.load(model_path, map_location=map_location)
            self.key_dim = model_weights['key_proj.key_proj.weight'].shape[0]
            self.value_dim = model_weights['value_encoder.fuser.block2.conv2.weight'].shape[0]
            self.disable_hidden = 'decoder.hidden_update.transform.weight' not in model_weights
            if self.disable_hidden:
                self.hidden_dim = 0
            else:
                self.hidden_dim = model_weights['decoder.hidden_update.transform.weight'].shape[0] // 3
            None
        else:
            model_weights = None
            if 'key_dim' not in config:
                self.key_dim = 64
                None
            else:
                self.key_dim = config['key_dim']
            if 'value_dim' not in config:
                self.value_dim = 512
                None
            else:
                self.value_dim = config['value_dim']
            if 'hidden_dim' not in config:
                self.hidden_dim = 64
                None
            else:
                self.hidden_dim = config['hidden_dim']
            self.disable_hidden = self.hidden_dim <= 0
        config['key_dim'] = self.key_dim
        config['value_dim'] = self.value_dim
        config['hidden_dim'] = self.hidden_dim
        return model_weights

    def load_weights(self, src_dict, init_as_zero_if_needed=False):
        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    None
                    pads = torch.zeros((64, 1, 7, 7), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        None
                        nn.init.orthogonal_(pads)
                    else:
                        None
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)
        self.load_state_dict(src_dict)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASPP,
     lambda: ([], {'in_channels': 4, 'atrous_rates': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ASPPConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ASPPPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (AtrousSeparableConvolution,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BRSMaskLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlockV1b,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchNorm2dNoSync,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BootstrappedCE,
     lambda: ([], {'start_warm': 4, 'end_warm': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), 0], {}),
     False),
    (CBAM,
     lambda: ([], {'gate_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChannelGate,
     lambda: ([], {'gate_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChannelPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DistMaps,
     lambda: ([], {'norm_radius': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GConv2D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 1, 4, 4])], {}),
     False),
    (GroupResBlock,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 1, 4, 4])], {}),
     False),
    (HighResolutionNet,
     lambda: ([], {'width': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ObjectAttentionBlock2D,
     lambda: ([], {'in_channels': 4, 'key_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResNetBackbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SpatialGate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialGather_Module,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialOCR_Module,
     lambda: ([], {'in_channels': 4, 'key_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (_ASPP,
     lambda: ([], {'in_channels': 4, 'atrous_rates': [4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_AsppPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'norm_layer': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_BatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_SimpleSegmentationModel,
     lambda: ([], {'backbone': _mock_layer(), 'classifier': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_hkchengrex_XMem(_paritybench_base):
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

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

