import sys
_module = sys.modules[__name__]
del sys
mix_sets = _module
eval = _module
pkg_usage = _module
run_coco_eval = _module
setup = _module
train = _module
yolact_edge = _module
backbone = _module
data = _module
coco = _module
config = _module
flying_chairs = _module
sampler_utils = _module
youtube_vis = _module
inference = _module
layers = _module
box_utils = _module
functions = _module
detection = _module
interpolate = _module
modules = _module
multibox_loss = _module
optical_flow_loss = _module
output_utils = _module
warp_utils = _module
augment_bbox = _module
bbox_recall = _module
cluster_bbox_sizes = _module
compute_masks = _module
convert_darknet = _module
make_grid = _module
optimize_bboxes = _module
parse_eval = _module
plot_loss = _module
save_bboxes = _module
unpack_statedict = _module
utils = _module
augmentations = _module
functions = _module
logging_helper = _module
merge_model = _module
misc = _module
tensorboard_helper = _module
tensorrt = _module
timer = _module
yolact = _module

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


import numpy as np


import torch


import torch.backends.cudnn as cudnn


from torch.autograd import Variable


import time


import random


from collections import defaultdict


from collections import OrderedDict


import matplotlib.pyplot as plt


import logging


import math


from queue import Queue


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import torch.utils.data as data


import torch.distributed as dist


import torch.multiprocessing as mp


from functools import partial


import torchvision.transforms as transforms


from math import sqrt


from torch.utils.data.sampler import Sampler


from torch.utils.data.sampler import BatchSampler


import itertools


from typing import List


from itertools import product


from numpy import random


from scipy.optimize import minimize


from torchvision import transforms


import types


from collections import deque


from torch.utils.tensorboard import SummaryWriter


import torchvision


from torchvision.models.resnet import Bottleneck


from torchvision.models.resnet import conv1x1


from torchvision.models.resnet import conv3x3


from itertools import chain


from typing import Tuple


from typing import Optional


from torch import Tensor


import copy


class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """

    def gather(self, outputs, output_device):
        return sum(outputs, [])


class Bottleneck(nn.Module):
    """ Adapted from torchvision.models.resnet """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, dilation=dilation)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = nn.Sequential()
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNetBackbone(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, layers, atrous_layers=[], block=Bottleneck, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self.norm_layer = norm_layer
        self.dilation = 1
        self.atrous_layers = atrous_layers
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._make_layer(block, 64, layers[0])
        self._make_layer(block, 128, layers[1], stride=2)
        self._make_layer(block, 256, layers[2], stride=2)
        self._make_layer(block, 512, layers[3], stride=2)
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(self, block, planes, blocks, stride=1):
        """ Here one layer means a string of n Bottleneck blocks. """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if len(self.layers) in self.atrous_layers:
                self.dilation += 1
                stride = 1
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, dilation=self.dilation), self.norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.norm_layer, self.dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer))
        layer = nn.Sequential(*layers)
        self.channels.append(planes * block.expansion)
        self.layers.append(layer)
        return layer

    def forward(self, x, partial: bool=False):
        """ Returns a list of convouts for each layer. """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        layer_idx = 0
        for layer in self.layers:
            layer_idx += 1
            if not partial or layer_idx <= 2:
                x = layer(x)
                outs.append(x)
        return outs

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path, map_location='cpu')
        keys = list(state_dict)
        for key in keys:
            if key.startswith('layer'):
                idx = int(key[5])
                new_key = 'layers.' + str(idx - 1) + key[6:]
                state_dict[new_key] = state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)

    def add_layer(self, conv_channels=1024, downsample=2, depth=1, block=Bottleneck):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(block, conv_channels // block.expansion, blocks=depth, stride=downsample)


class ResNetBackboneGN(ResNetBackbone):

    def __init__(self, layers, num_groups=32):
        super().__init__(layers, norm_layer=lambda x: nn.GroupNorm(num_groups, x))

    def init_backbone(self, path):
        """ The path here comes from detectron. So we load it differently. """
        with open(path, 'rb') as f:
            state_dict = pickle.load(f, encoding='latin1')
            state_dict = state_dict['blobs']
        our_state_dict_keys = list(self.state_dict().keys())
        new_state_dict = {}
        gn_trans = lambda x: 'gn_s' if x == 'weight' else 'gn_b'
        layeridx2res = lambda x: 'res' + str(int(x) + 2)
        block2branch = lambda x: 'branch2' + ('a', 'b', 'c')[int(x[-1:]) - 1]
        for key in our_state_dict_keys:
            parts = key.split('.')
            transcribed_key = ''
            if parts[0] == 'conv1':
                transcribed_key = 'conv1_w'
            elif parts[0] == 'bn1':
                transcribed_key = 'conv1_' + gn_trans(parts[1])
            elif parts[0] == 'layers':
                if int(parts[1]) >= self.num_base_layers:
                    continue
                transcribed_key = layeridx2res(parts[1])
                transcribed_key += '_' + parts[2] + '_'
                if parts[3] == 'downsample':
                    transcribed_key += 'branch1_'
                    if parts[4] == '0':
                        transcribed_key += 'w'
                    else:
                        transcribed_key += gn_trans(parts[5])
                else:
                    transcribed_key += block2branch(parts[3]) + '_'
                    if 'conv' in parts[3]:
                        transcribed_key += 'w'
                    else:
                        transcribed_key += gn_trans(parts[4])
            new_state_dict[key] = torch.Tensor(state_dict[transcribed_key])
        self.load_state_dict(new_state_dict, strict=False)


def darknetconvlayer(in_channels, out_channels, *args, **kwdargs):
    """
    Implements a conv, activation, then batch norm.
    Arguments are passed into the conv layer.
    """
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, *args, **kwdargs, bias=False), nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.1, inplace=True))


class DarkNetBlock(nn.Module):
    """ Note: channels is the lesser of the two. The output will be expansion * channels. """
    expansion = 2

    def __init__(self, in_channels, channels):
        super().__init__()
        self.conv1 = darknetconvlayer(in_channels, channels, kernel_size=1)
        self.conv2 = darknetconvlayer(channels, channels * self.expansion, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x


class DarkNetBackbone(nn.Module):
    """
    An implementation of YOLOv3's Darnet53 in
    https://pjreddie.com/media/files/papers/YOLOv3.pdf

    This is based off of the implementation of Resnet above.
    """

    def __init__(self, layers=[1, 2, 8, 8, 4], block=DarkNetBlock):
        super().__init__()
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self._preconv = darknetconvlayer(3, 32, kernel_size=3, padding=1)
        self.in_channels = 32
        self._make_layer(block, 32, layers[0])
        self._make_layer(block, 64, layers[1])
        self._make_layer(block, 128, layers[2])
        self._make_layer(block, 256, layers[3])
        self._make_layer(block, 512, layers[4])
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(self, block, channels, num_blocks, stride=2):
        """ Here one layer means a string of n blocks. """
        layer_list = []
        layer_list.append(darknetconvlayer(self.in_channels, channels * block.expansion, kernel_size=3, padding=1, stride=stride))
        self.in_channels = channels * block.expansion
        layer_list += [block(self.in_channels, channels) for _ in range(num_blocks)]
        self.channels.append(self.in_channels)
        self.layers.append(nn.Sequential(*layer_list))

    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        x = self._preconv(x)
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        return tuple(outs)

    def add_layer(self, conv_channels=1024, stride=2, depth=1, block=DarkNetBlock):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(block, conv_channels // block.expansion, num_blocks=depth, stride=stride)

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        self.load_state_dict(torch.load(path, map_location='cpu'), strict=False)


class VGGBackbone(nn.Module):
    """
    Args:
        - cfg: A list of layers given as lists. Layers can be either 'M' signifying
                a max pooling layer, a number signifying that many feature maps in
                a conv layer, or a tuple of 'M' or a number and a kwdargs dict to pass
                into the function that creates the layer (e.g. nn.MaxPool2d for 'M').
        - extra_args: A list of lists of arguments to pass into add_layer.
        - norm_layers: Layers indices that need to pass through an l2norm layer.
    """

    def __init__(self, cfg, extra_args=[], norm_layers=[]):
        super().__init__()
        self.channels = []
        self.layers = nn.ModuleList()
        self.in_channels = 3
        self.extra_args = list(reversed(extra_args))
        self.total_layer_count = 0
        self.state_dict_lookup = {}
        for idx, layer_cfg in enumerate(cfg):
            self._make_layer(layer_cfg)
        self.norms = nn.ModuleList([nn.BatchNorm2d(self.channels[l]) for l in norm_layers])
        self.norm_lookup = {l: idx for idx, l in enumerate(norm_layers)}
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(self, cfg):
        """
        Each layer is a sequence of conv layers usually preceded by a max pooling.
        Adapted from torchvision.models.vgg.make_layers.
        """
        layers = []
        for v in cfg:
            args = None
            if isinstance(v, tuple):
                args = v[1]
                v = v[0]
            if v == 'M':
                if args is None:
                    args = {'kernel_size': 2, 'stride': 2}
                layers.append(nn.MaxPool2d(**args))
            else:
                cur_layer_idx = self.total_layer_count + len(layers)
                self.state_dict_lookup[cur_layer_idx] = '%d.%d' % (len(self.layers), len(layers))
                if args is None:
                    args = {'kernel_size': 3, 'padding': 1}
                layers.append(nn.Conv2d(self.in_channels, v, **args))
                layers.append(nn.ReLU(inplace=True))
                self.in_channels = v
        self.total_layer_count += len(layers)
        self.channels.append(self.in_channels)
        self.layers.append(nn.Sequential(*layers))

    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        outs = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx in self.norm_lookup:
                x = self.norms[self.norm_lookup[idx]](x)
            outs.append(x)
        return tuple(outs)

    def transform_key(self, k):
        """ Transform e.g. features.24.bias to layers.4.1.bias """
        vals = k.split('.')
        layerIdx = self.state_dict_lookup[int(vals[0])]
        return 'layers.%s.%s' % (layerIdx, vals[1])

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path, map_location='cpu')
        state_dict = OrderedDict([(self.transform_key(k), v) for k, v in state_dict.items()])
        self.load_state_dict(state_dict, strict=False)

    def add_layer(self, conv_channels=128, downsample=2):
        """ Add a downsample layer to the backbone as per what SSD does. """
        if len(self.extra_args) > 0:
            conv_channels, downsample = self.extra_args.pop()
        padding = 1 if downsample > 1 else 0
        layer = nn.Sequential(nn.Conv2d(self.in_channels, conv_channels, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, stride=downsample, padding=padding), nn.ReLU(inplace=True))
        self.in_channels = conv_channels * 2
        self.channels.append(self.in_channels)
        self.layers.append(layer)


class ConvBNAct(nn.Sequential):
    """
    Adapted from torchvision.models.mobilenet.ConvBNReLU
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, activation=nn.ReLU6(inplace=True)):
        padding = (kernel_size - 1) // 2
        super(ConvBNAct, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False), nn.BatchNorm2d(out_planes), activation)


ConvBNReLU = partial(ConvBNAct)


class InvertedResidual(nn.Module):
    """
    Adapted from torchvision.models.mobilenet.InvertedResidual
    """

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def _make_divisible(v, divisor, min_value=None):
    """
    Adapted from torchvision.models.mobilenet._make_divisible.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV2Backbone(nn.Module):
    """
    Adapted from torchvision.models.mobilenet.MobileNetV2
    """

    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, block=InvertedResidual):
        super(MobileNetV2Backbone, self).__init__()
        input_channel = 32
        last_channel = 1280
        self.channels = []
        self.layers = nn.ModuleList()
        if inverted_residual_setting is None:
            raise ValueError('Must provide inverted_residual_setting where each element is a list that represents the MobileNetV2 t,c,n,s values for that layer.')
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError('inverted_residual_setting should be non-empty or a 4-element list, got {}'.format(inverted_residual_setting))
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.layers.append(ConvBNReLU(3, input_channel, stride=2))
        self.channels.append(input_channel)
        for t, c, n, s in inverted_residual_setting:
            input_channel = self._make_layer(input_channel, width_mult, round_nearest, t, c, n, s, block)
            self.channels.append(input_channel)
        self.layers.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.channels.append(self.last_channel)
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(self, input_channel, width_mult, round_nearest, t, c, n, s, block):
        """A layer is a combination of inverted residual blocks"""
        layers = []
        output_channel = _make_divisible(c * width_mult, round_nearest)
        for i in range(n):
            stride = s if i == 0 else 1
            layers.append(block(input_channel, output_channel, stride, expand_ratio=t))
            input_channel = output_channel
        self.layers.append(nn.Sequential(*layers))
        return input_channel

    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        outs = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            outs.append(x)
        return tuple(outs)

    def add_layer(self, conv_channels=1280, t=1, c=1280, n=1, s=2):
        """TODO: Need to make sure that this works as intended.
        """
        self._make_layer(conv_channels, 1.0, 8, t, c, n, s, InvertedResidual)

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        checkpoint = torch.load(path)
        checkpoint.pop('classifier.1.weight')
        checkpoint.pop('classifier.1.bias')
        checkpoint_keys = list(checkpoint.keys())
        assert len(checkpoint_keys) == len(self.state_dict())
        transform_dict = dict(zip(checkpoint, list(self.state_dict().keys())))
        state_dict = OrderedDict([(transform_dict[k], v) for k, v in checkpoint.items()])
        self.load_state_dict(state_dict, strict=True)


class InterpolateModule(nn.Module):
    """
	This is a module version of F.interpolate (rip nn.Upsampling).
	Any arguments you give it just get passed along for the ride.
	"""

    def __init__(self, *args, **kwdargs):
        super().__init__()
        self.args = args
        self.kwdargs = kwdargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwdargs)


class Config(object):
    """
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """
        ret = Config(vars(self))
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)
        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)
        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def print(self):
        for k, v in vars(self).items():
            None


activation_func = Config({'tanh': torch.tanh, 'sigmoid': torch.sigmoid, 'softmax': lambda x: torch.nn.functional.softmax(x, dim=-1), 'relu': lambda x: torch.nn.functional.relu(x, inplace=True), 'none': lambda x: x})


@torch.jit.script
def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2]), 1)


@torch.jit.script
def sanitize_coordinates(_x1, _x2, img_size: int, padding: int=0, cast: bool=True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1 - padding, min=0)
    x2 = torch.clamp(x2 + padding, max=img_size)
    return x1, x2


@torch.jit.script
def crop(masks, boxes, padding: int=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)
    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)
    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)
    crop_mask = masks_left * masks_right * masks_up * masks_down
    return masks * crop_mask.float()


@torch.jit.script
def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)


@torch.jit.script
def decode(loc, priors, use_yolo_regressors: bool=False):
    """
    Decode predicted bbox coordinates using the same scheme
    employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf

        b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
        b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
        b_w = prior_w * exp(loc_w)
        b_h = prior_h * exp(loc_h)
    
    Note that loc is inputed as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
    while priors are inputed as [x, y, w, h] where each coordinate
    is relative to size of the image (even sigmoid(x)). We do this
    in the network by dividing by the 'cell size', which is just
    the size of the convouts.
    
    Also note that prior_x and prior_y are center coordinates which
    is why we have to subtract .5 from sigmoid(pred_x and pred_y).
    
    Args:
        - loc:    The predicted bounding boxes of size [num_priors, 4]
        - priors: The priorbox coords with size [num_priors, 4]
    
    Returns: A tensor of decoded relative coordinates in point form 
             form with size [num_priors, 4]
    """
    if use_yolo_regressors:
        boxes = torch.cat((loc[:, :2] + priors[:, :2], priors[:, 2:] * torch.exp(loc[:, 2:])), 1)
        boxes = point_form(boxes)
    else:
        variances = [0.1, 0.2]
        boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1)) + x_max


mask_type = Config({'direct': 0, 'lincomb': 1})


def change(gt, priors):
    """
    Compute the d_change metric proposed in Box2Pix:
    https://lmb.informatik.uni-freiburg.de/Publications/2018/UB18/paper-box2pix.pdf
    
    Input should be in point form (xmin, ymin, xmax, ymax).

    Output is of shape [num_gt, num_priors]
    Note this returns -change so it can be a drop in replacement for 
    """
    num_priors = priors.size(0)
    num_gt = gt.size(0)
    gt_w = (gt[:, 2] - gt[:, 0])[:, None].expand(num_gt, num_priors)
    gt_h = (gt[:, 3] - gt[:, 1])[:, None].expand(num_gt, num_priors)
    gt_mat = gt[:, None, :].expand(num_gt, num_priors, 4)
    pr_mat = priors[None, :, :].expand(num_gt, num_priors, 4)
    diff = gt_mat - pr_mat
    diff[:, :, 0] /= gt_w
    diff[:, :, 2] /= gt_w
    diff[:, :, 1] /= gt_h
    diff[:, :, 3] /= gt_h
    return -torch.sqrt((diff ** 2).sum(dim=2))


@torch.jit.script
def encode(matched, priors, use_yolo_regressors: bool=False):
    """
    Encode bboxes matched with each prior into the format
    produced by the network. See decode for more details on
    this format. Note that encode(decode(x, p), p) = x.
    
    Args:
        - matched: A tensor of bboxes in point form with shape [num_priors, 4]
        - priors:  The tensor of all priors with shape [num_priors, 4]
    Return: A tensor with encoded relative coordinates in the format
            outputted by the network (see decode). Size: [num_priors, 4]
    """
    if use_yolo_regressors:
        boxes = center_size(matched)
        loc = torch.cat((boxes[:, :2] - priors[:, :2], torch.log(boxes[:, 2:] / priors[:, 2:])), 1)
    else:
        variances = [0.1, 0.2]
        g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
        g_cxcy /= variances[0] * priors[:, 2:]
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = torch.log(g_wh) / variances[1]
        loc = torch.cat([g_cxcy, g_wh], 1)
    return loc


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip(max_xy - min_xy, a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard(box_a, box_b, iscrowd=False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    if iscrowd:
        return inter / area_a
    else:
        return inter / union


def match(pos_thresh, neg_thresh, truths, priors, labels, crowd_boxes, loc_t, conf_t, idx_t, idx, loc_data):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        pos_thresh: (float) IoU > pos_thresh ==> positive.
        neg_thresh: (float) IoU < neg_thresh ==> negative.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        crowd_boxes: (tensor) All the crowd box annotations or None if there are none.
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds. Note: -1 means neutral.
        idx_t: (tensor) Tensor to be filled w/ the index of the matched gt box for each prior.
        idx: (int) current batch index.
        loc_data: (tensor) The predicted bbox regression coordinates for this batch.
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    decoded_priors = decode(loc_data, priors, cfg.use_yolo_regressors) if cfg.use_prediction_matching else point_form(priors)
    overlaps = jaccard(truths, decoded_priors) if not cfg.use_change_matching else change(truths, decoded_priors)
    best_truth_overlap, best_truth_idx = overlaps.max(0)
    for _ in range(overlaps.size(0)):
        best_prior_overlap, best_prior_idx = overlaps.max(1)
        j = best_prior_overlap.max(0)[1]
        i = best_prior_idx[j]
        overlaps[:, i] = -1
        overlaps[j, :] = -1
        best_truth_overlap[i] = 2
        best_truth_idx[i] = j
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx] + 1
    conf[best_truth_overlap < pos_thresh] = -1
    conf[best_truth_overlap < neg_thresh] = 0
    if crowd_boxes is not None and cfg.crowd_iou_threshold < 1:
        crowd_overlaps = jaccard(decoded_priors, crowd_boxes, iscrowd=True)
        best_crowd_overlap, best_crowd_idx = crowd_overlaps.max(1)
        conf[(conf <= 0) & (best_crowd_overlap > cfg.crowd_iou_threshold)] = -1
    loc = encode(matches, priors, cfg.use_yolo_regressors)
    loc_t[idx] = loc
    conf_t[idx] = conf
    idx_t[idx] = best_truth_idx


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, pos_threshold, neg_threshold, negpos_ratio):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.negpos_ratio = negpos_ratio
        self.l1_expected_area = 20 * 20 / 70 / 70
        self.l1_alpha = 0.1

    def forward(self, predictions, targets, masks, num_crowds):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            mask preds, and prior boxes from SSD net.
                loc shape: torch.size(batch_size,num_priors,4)
                conf shape: torch.size(batch_size,num_priors,num_classes)
                masks shape: torch.size(batch_size,num_priors,mask_dim)
                priors shape: torch.size(num_priors,4)
                proto* shape: torch.size(batch_size,mask_h,mask_w,mask_dim)

            targets (list<tensor>): Ground truth boxes and labels for a batch,
                shape: [batch_size][num_objs,5] (last idx is the label).

            masks (list<tensor>): Ground truth masks for each object in each image,
                shape: [batch_size][num_objs,im_height,im_width]

            num_crowds (list<int>): Number of crowd annotations per batch. The crowd
                annotations should be the last num_crowds elements of targets and masks.
            
            * Only if mask_type == lincomb
        """
        loc_data = predictions['loc']
        conf_data = predictions['conf']
        mask_data = predictions['mask']
        priors = predictions['priors']
        if cfg.mask_type == mask_type.lincomb:
            proto_data = predictions['proto']
        if cfg.use_instance_coeff:
            inst_data = predictions['inst']
        else:
            inst_data = None
        labels = [None] * len(targets)
        batch_size = loc_data.size(0)
        num_priors = priors.size(0)
        num_classes = self.num_classes
        loc_t = loc_data.new(batch_size, num_priors, 4)
        gt_box_t = loc_data.new(batch_size, num_priors, 4)
        conf_t = loc_data.new(batch_size, num_priors).long()
        idx_t = loc_data.new(batch_size, num_priors).long()
        defaults = priors.data
        if cfg.use_class_existence_loss:
            class_existence_t = loc_data.new(batch_size, num_classes - 1)
        for idx in range(batch_size):
            truths = targets[idx][:, :-1].data
            labels[idx] = targets[idx][:, -1].data.long()
            if cfg.use_class_existence_loss:
                class_existence_t[idx, :] = torch.eye(num_classes - 1, device=conf_t.get_device())[labels[idx]].max(dim=0)[0]
            cur_crowds = num_crowds[idx]
            if cur_crowds > 0:
                split = lambda x: (x[-cur_crowds:], x[:-cur_crowds])
                crowd_boxes, truths = split(truths)
                _, labels[idx] = split(labels[idx])
                _, masks[idx] = split(masks[idx])
            else:
                crowd_boxes = None
            match(self.pos_threshold, self.neg_threshold, truths, defaults, labels[idx], crowd_boxes, loc_t, conf_t, idx_t, idx, loc_data[idx])
            gt_box_t[idx, :, :] = truths[idx_t[idx]]
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        idx_t = Variable(idx_t, requires_grad=False)
        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        losses = {}
        if cfg.train_boxes:
            loc_p = loc_data[pos_idx].view(-1, 4)
            loc_t = loc_t[pos_idx].view(-1, 4)
            losses['B'] = F.smooth_l1_loss(loc_p, loc_t, reduction='sum') * cfg.bbox_alpha
        if cfg.train_masks:
            if cfg.mask_type == mask_type.direct:
                if cfg.use_gt_bboxes:
                    pos_masks = []
                    for idx in range(batch_size):
                        pos_masks.append(masks[idx][idx_t[idx, pos[idx]]])
                    masks_t = torch.cat(pos_masks, 0)
                    masks_p = mask_data[pos, :].view(-1, cfg.mask_dim)
                    losses['M'] = F.binary_cross_entropy(torch.clamp(masks_p, 0, 1), masks_t, reduction='sum') * cfg.mask_alpha
                else:
                    losses['M'] = self.direct_mask_loss(pos_idx, idx_t, loc_data, mask_data, priors, masks)
            elif cfg.mask_type == mask_type.lincomb:
                losses.update(self.lincomb_mask_loss(pos, idx_t, loc_data, mask_data, priors, proto_data, masks, gt_box_t, inst_data))
                if cfg.mask_proto_loss is not None:
                    if cfg.mask_proto_loss == 'l1':
                        losses['P'] = torch.mean(torch.abs(proto_data)) / self.l1_expected_area * self.l1_alpha
                    elif cfg.mask_proto_loss == 'disj':
                        losses['P'] = -torch.mean(torch.max(F.log_softmax(proto_data, dim=-1), dim=-1)[0])
        if cfg.use_focal_loss:
            if cfg.use_sigmoid_focal_loss:
                losses['C'] = self.focal_conf_sigmoid_loss(conf_data, conf_t)
            elif cfg.use_objectness_score:
                losses['C'] = self.focal_conf_objectness_loss(conf_data, conf_t)
            else:
                losses['C'] = self.focal_conf_loss(conf_data, conf_t)
        else:
            losses['C'] = self.ohem_conf_loss(conf_data, conf_t, pos, batch_size)
        if cfg.use_class_existence_loss:
            losses['E'] = self.class_existence_loss(predictions['classes'], class_existence_t)
        if cfg.use_semantic_segmentation_loss:
            losses['S'] = self.semantic_segmentation_loss(predictions['segm'], masks, labels)
        total_num_pos = num_pos.data.sum().float()
        for k in losses:
            if k not in ('P', 'E', 'S'):
                losses[k] /= total_num_pos
            else:
                losses[k] /= batch_size
        return losses

    def class_existence_loss(self, class_data, class_existence_t):
        return cfg.class_existence_alpha * F.binary_cross_entropy_with_logits(class_data, class_existence_t, reduction='sum')

    def semantic_segmentation_loss(self, segment_data, mask_t, class_t, interpolation_mode='bilinear'):
        batch_size, num_classes, mask_h, mask_w = segment_data.size()
        loss_s = 0
        for idx in range(batch_size):
            cur_segment = segment_data[idx]
            cur_class_t = class_t[idx]
            with torch.no_grad():
                downsampled_masks = F.interpolate(mask_t[idx].unsqueeze(0), (mask_h, mask_w), mode=interpolation_mode, align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.gt(0.5).float()
                segment_t = torch.zeros_like(cur_segment, requires_grad=False)
                for obj_idx in range(downsampled_masks.size(0)):
                    segment_t[cur_class_t[obj_idx]] = torch.max(segment_t[cur_class_t[obj_idx]], downsampled_masks[obj_idx])
            loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_t, reduction='sum')
        return loss_s / mask_h / mask_w * cfg.semantic_segmentation_alpha

    def ohem_conf_loss(self, conf_data, conf_t, pos, num):
        batch_conf = conf_data.view(-1, self.num_classes)
        if cfg.ohem_use_most_confident:
            batch_conf = F.softmax(batch_conf, dim=1)
            loss_c, _ = batch_conf[:, 1:].max(dim=1)
        else:
            loss_c = log_sum_exp(batch_conf) - batch_conf[:, 0]
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0
        loss_c[conf_t < 0] = 0
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        neg[pos] = 0
        neg[conf_t < 0] = 0
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
        return cfg.conf_alpha * loss_c

    def focal_conf_loss(self, conf_data, conf_t):
        """
        Focal loss as described in https://arxiv.org/pdf/1708.02002.pdf
        Adapted from https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
        Note that this uses softmax and not the original sigmoid from the paper.
        """
        conf_t = conf_t.view(-1)
        conf_data = conf_data.view(-1, conf_data.size(-1))
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0
        logpt = F.log_softmax(conf_data, dim=-1)
        logpt = logpt.gather(1, conf_t.unsqueeze(-1))
        logpt = logpt.view(-1)
        pt = logpt.exp()
        background = (conf_t == 0).float()
        at = (1 - cfg.focal_loss_alpha) * background + cfg.focal_loss_alpha * (1 - background)
        loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt
        return cfg.conf_alpha * (loss * keep).sum()

    def focal_conf_sigmoid_loss(self, conf_data, conf_t):
        """
        Focal loss but using sigmoid like the original paper.
        Note: To make things mesh easier, the network still predicts 81 class confidences in this mode.
              Because retinanet originally only predicts 80, we simply just don't use conf_data[..., 0]
        """
        num_classes = conf_data.size(-1)
        conf_t = conf_t.view(-1)
        conf_data = conf_data.view(-1, num_classes)
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0
        conf_one_t = torch.eye(num_classes, device=conf_t.get_device())[conf_t]
        conf_pm_t = conf_one_t * 2 - 1
        logpt = F.logsigmoid(conf_data * conf_pm_t)
        pt = logpt.exp()
        at = cfg.focal_loss_alpha * conf_one_t + (1 - cfg.focal_loss_alpha) * (1 - conf_one_t)
        at[..., 0] = 0
        loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt
        loss = keep * loss.sum(dim=-1)
        return cfg.conf_alpha * loss.sum()

    def focal_conf_objectness_loss(self, conf_data, conf_t):
        """
        Instead of using softmax, use class[0] to be the objectness score and do sigmoid focal loss on that.
        Then for the rest of the classes, softmax them and apply CE for only the positive examples.

        If class[0] = 1 implies forground and class[0] = 0 implies background then you achieve something
        similar during test-time to softmax by setting class[1:] = softmax(class[1:]) * class[0] and invert class[0].
        """
        conf_t = conf_t.view(-1)
        conf_data = conf_data.view(-1, conf_data.size(-1))
        keep = (conf_t >= 0).float()
        conf_t[conf_t < 0] = 0
        background = (conf_t == 0).float()
        at = (1 - cfg.focal_loss_alpha) * background + cfg.focal_loss_alpha * (1 - background)
        logpt = F.logsigmoid(conf_data[:, 0]) * (1 - background) + F.logsigmoid(-conf_data[:, 0]) * background
        pt = logpt.exp()
        obj_loss = -at * (1 - pt) ** cfg.focal_loss_gamma * logpt
        pos_mask = conf_t > 0
        conf_data_pos = conf_data[:, 1:][pos_mask]
        conf_t_pos = conf_t[pos_mask] - 1
        class_loss = F.cross_entropy(conf_data_pos, conf_t_pos, reduction='sum')
        return cfg.conf_alpha * (class_loss + (obj_loss * keep).sum())

    def direct_mask_loss(self, pos_idx, idx_t, loc_data, mask_data, priors, masks):
        """ Crops the gt masks using the predicted bboxes, scales them down, and outputs the BCE loss. """
        loss_m = 0
        for idx in range(mask_data.size(0)):
            with torch.no_grad():
                cur_pos_idx = pos_idx[idx, :, :]
                cur_pos_idx_squeezed = cur_pos_idx[:, 1]
                pos_bboxes = decode(loc_data[idx, :, :], priors.data, cfg.use_yolo_regressors)
                pos_bboxes = pos_bboxes[cur_pos_idx].view(-1, 4).clamp(0, 1)
                pos_lookup = idx_t[idx, cur_pos_idx_squeezed]
                cur_masks = masks[idx]
                pos_masks = cur_masks[pos_lookup, :, :]
                num_pos, img_height, img_width = pos_masks.size()
                x1, x2 = sanitize_coordinates(pos_bboxes[:, 0], pos_bboxes[:, 2], img_width)
                y1, y2 = sanitize_coordinates(pos_bboxes[:, 1], pos_bboxes[:, 3], img_height)
                scaled_masks = []
                for jdx in range(num_pos):
                    tmp_mask = pos_masks[jdx, y1[jdx]:y2[jdx], x1[jdx]:x2[jdx]]
                    while tmp_mask.dim() < 2:
                        tmp_mask = tmp_mask.unsqueeze(0)
                    new_mask = F.adaptive_avg_pool2d(tmp_mask.unsqueeze(0), cfg.mask_size)
                    scaled_masks.append(new_mask.view(1, -1))
                mask_t = torch.cat(scaled_masks, 0).gt(0.5).float()
            pos_mask_data = mask_data[idx, cur_pos_idx_squeezed, :]
            loss_m += F.binary_cross_entropy(torch.clamp(pos_mask_data, 0, 1), mask_t, reduction='sum') * cfg.mask_alpha
        return loss_m

    def coeff_diversity_loss(self, coeffs, instance_t):
        """
        coeffs     should be size [num_pos, num_coeffs]
        instance_t should be size [num_pos] and be values from 0 to num_instances-1
        """
        num_pos = coeffs.size(0)
        instance_t = instance_t.view(-1)
        coeffs_norm = F.normalize(coeffs, dim=1)
        cos_sim = coeffs_norm @ coeffs_norm.t()
        inst_eq = (instance_t[:, None].expand_as(cos_sim) == instance_t[None, :].expand_as(cos_sim)).float()
        cos_sim = (cos_sim + 1) / 2
        loss = (1 - cos_sim) * inst_eq + cos_sim * (1 - inst_eq)
        return cfg.mask_proto_coeff_diversity_alpha * loss.sum() / num_pos

    def lincomb_mask_loss(self, pos, idx_t, loc_data, mask_data, priors, proto_data, masks, gt_box_t, inst_data, interpolation_mode='bilinear'):
        mask_h = proto_data.size(1)
        mask_w = proto_data.size(2)
        process_gt_bboxes = cfg.mask_proto_normalize_emulate_roi_pooling or cfg.mask_proto_crop
        if cfg.mask_proto_remove_empty_masks:
            pos = pos.clone()
        loss_m = 0
        loss_d = 0
        for idx in range(mask_data.size(0)):
            with torch.no_grad():
                downsampled_masks = F.interpolate(masks[idx].unsqueeze(0), (mask_h, mask_w), mode=interpolation_mode, align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()
                if cfg.mask_proto_binarize_downsampled_gt:
                    downsampled_masks = downsampled_masks.gt(0.5).float()
                if cfg.mask_proto_remove_empty_masks:
                    very_small_masks = downsampled_masks.sum(dim=(0, 1)) <= 0.0001
                    for i in range(very_small_masks.size(0)):
                        if very_small_masks[i]:
                            pos[idx, idx_t[idx] == i] = 0
                if cfg.mask_proto_reweight_mask_loss:
                    if not cfg.mask_proto_binarize_downsampled_gt:
                        bin_gt = downsampled_masks.gt(0.5).float()
                    else:
                        bin_gt = downsampled_masks
                    gt_foreground_norm = bin_gt / (torch.sum(bin_gt, dim=(0, 1), keepdim=True) + 0.0001)
                    gt_background_norm = (1 - bin_gt) / (torch.sum(1 - bin_gt, dim=(0, 1), keepdim=True) + 0.0001)
                    mask_reweighting = gt_foreground_norm * cfg.mask_proto_reweight_coeff + gt_background_norm
                    mask_reweighting *= mask_h * mask_w
            cur_pos = pos[idx]
            pos_idx_t = idx_t[idx, cur_pos]
            if process_gt_bboxes:
                pos_gt_box_t = gt_box_t[idx, cur_pos]
            if pos_idx_t.size(0) == 0:
                continue
            proto_masks = proto_data[idx]
            proto_coef = mask_data[idx, cur_pos, :]
            if cfg.mask_proto_coeff_diversity_loss:
                if inst_data is not None:
                    div_coeffs = inst_data[idx, cur_pos, :]
                else:
                    div_coeffs = proto_coef
                loss_d += self.coeff_diversity_loss(div_coeffs, pos_idx_t)
            old_num_pos = proto_coef.size(0)
            if old_num_pos > cfg.masks_to_train:
                perm = torch.randperm(proto_coef.size(0))
                select = perm[:cfg.masks_to_train]
                proto_coef = proto_coef[select, :]
                pos_idx_t = pos_idx_t[select]
                if process_gt_bboxes:
                    pos_gt_box_t = pos_gt_box_t[select, :]
            num_pos = proto_coef.size(0)
            mask_t = downsampled_masks[:, :, pos_idx_t]
            pred_masks = proto_masks @ proto_coef.t()
            pred_masks = cfg.mask_proto_mask_activation(pred_masks)
            if cfg.mask_proto_double_loss:
                if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                    pre_loss = F.binary_cross_entropy(torch.clamp(pred_masks, 0, 1), mask_t, reduction='sum')
                else:
                    pre_loss = F.smooth_l1_loss(pred_masks, mask_t, reduction='sum')
                loss_m += cfg.mask_proto_double_loss_alpha * pre_loss
            if cfg.mask_proto_crop:
                pred_masks = crop(pred_masks, pos_gt_box_t)
            if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                pre_loss = F.binary_cross_entropy(torch.clamp(pred_masks, 0, 1), mask_t, reduction='none')
            else:
                pre_loss = F.smooth_l1_loss(pred_masks, mask_t, reduction='none')
            if cfg.mask_proto_normalize_mask_loss_by_sqrt_area:
                gt_area = torch.sum(mask_t, dim=(0, 1), keepdim=True)
                pre_loss = pre_loss / (torch.sqrt(gt_area) + 0.0001)
            if cfg.mask_proto_reweight_mask_loss:
                pre_loss = pre_loss * mask_reweighting[:, :, pos_idx_t]
            if cfg.mask_proto_normalize_emulate_roi_pooling:
                weight = mask_h * mask_w if cfg.mask_proto_crop else 1
                pos_get_csize = center_size(pos_gt_box_t)
                gt_box_width = pos_get_csize[:, 2] * mask_w
                gt_box_height = pos_get_csize[:, 3] * mask_h
                pre_loss = pre_loss.sum(dim=(0, 1)) / gt_box_width / gt_box_height * weight
            if old_num_pos > num_pos:
                pre_loss *= old_num_pos / num_pos
            loss_m += torch.sum(pre_loss)
        losses = {'M': loss_m * cfg.mask_alpha / mask_h / mask_w}
        if cfg.mask_proto_coeff_diversity_loss:
            losses['D'] = loss_d
        return losses


class OpticalFlowLoss(nn.Module):

    def __init__(self):
        super(OpticalFlowLoss, self).__init__()

    def forward(self, preds, gt):
        losses = {}
        loss_F = 0
        for pred in preds:
            _, _, h, w = pred.size()
            gt_downsample = F.interpolate(gt, size=(h, w), mode='bilinear', align_corners=False)
            loss_F += torch.norm(pred - gt_downsample, dim=1).mean()
        losses['F'] = loss_F
        return losses


MEANS = 103.94, 116.78, 123.68


STD = 57.38, 57.12, 58.4


class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for super speed.
    This doesn't suppport a lot of config settings and should only be used for production.
    Maintain this as necessary.
    """

    def __init__(self):
        super().__init__()
        self.mean = torch.Tensor(MEANS).float()[None, :, None, None]
        self.std = torch.Tensor(STD).float()[None, :, None, None]
        self.transform = cfg.backbone.transform

    def forward(self, img):
        self.mean = self.mean
        self.std = self.std
        if cfg.preserve_aspect_ratio:
            raise NotImplementedError
        img = img.permute(0, 3, 1, 2).contiguous()
        if type(cfg.max_size) == tuple:
            img = F.interpolate(img, cfg.max_size[::-1], mode='bilinear', align_corners=False)
        else:
            img = F.interpolate(img, (cfg.max_size, cfg.max_size), mode='bilinear', align_corners=False)
        if self.transform.normalize:
            img = (img - self.mean) / self.std
        elif self.transform.subtract_means:
            img = img - self.mean
        elif self.transform.to_float:
            img = img / 255
        if self.transform.channel_order != 'RGB':
            raise NotImplementedError
        img = img[:, (2, 1, 0), :, :].contiguous()
        return img


class Concat(nn.Module):

    def __init__(self, nets, extra_params):
        super().__init__()
        self.nets = nn.ModuleList(nets)
        self.extra_params = extra_params

    def forward(self, x):
        return torch.cat([net(x) for net in self.nets], dim=1, **self.extra_params)


def make_net(in_channels, conf, include_last_relu=True):
    """
    A helper function to take a config setting and turn it into a network.
    Used by protonet and extrahead. Returns (network, out_channels)
    """

    def make_layer(layer_cfg):
        nonlocal in_channels
        if isinstance(layer_cfg[0], str):
            layer_name = layer_cfg[0]
            if layer_name == 'cat':
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
                layer = Concat([net[0] for net in nets], layer_cfg[2])
                num_channels = sum([net[1] for net in nets])
        else:
            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]
            if kernel_size > 0:
                layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])
            elif num_channels is None:
                layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False, **layer_cfg[2])
            else:
                layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])
        in_channels = num_channels if num_channels is not None else in_channels
        return [layer, nn.ReLU(inplace=True)]
    net = sum([make_layer(x) for x in conf], [])
    if not include_last_relu:
        net = net[:-1]
    return nn.Sequential(*net), in_channels


class PredictionModule(nn.Module):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.

    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    """

    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None, index=0):
        super().__init__()
        self.params = [in_channels, out_channels, aspect_ratios, scales, parent, index]
        self.num_classes = cfg.num_classes
        self.mask_dim = cfg.mask_dim
        self.num_priors = sum(len(x) for x in aspect_ratios)
        self.parent = [parent]
        self.index = index
        if cfg.mask_proto_prototypes_as_features:
            in_channels += self.mask_dim
        if parent is None:
            if cfg.extra_head_net is None:
                out_channels = in_channels
            else:
                self.upfeature, out_channels = make_net(in_channels, cfg.extra_head_net)
            if cfg.use_prediction_module:
                self.block = Bottleneck(out_channels, out_channels // 4)
                self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
                self.bn = nn.BatchNorm2d(out_channels)
            self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4, **cfg.head_layer_params)
            self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, **cfg.head_layer_params)
            self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, **cfg.head_layer_params)
            if cfg.use_instance_coeff:
                self.inst_layer = nn.Conv2d(out_channels, self.num_priors * cfg.num_instance_coeffs, **cfg.head_layer_params)

            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    return nn.Sequential(*sum([[nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True)] for _ in range(num_layers)], []))
            self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in cfg.extra_layers]
            if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_coeff_gate:
                self.gate_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, kernel_size=3, padding=1)
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.priors = None
        self.last_conv_size = None

    def forward(self, x):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        src = self if self.parent[0] is None else self.parent[0]
        conv_h = x.size(2)
        conv_w = x.size(3)
        if cfg.extra_head_net is not None:
            x = src.upfeature(x)
        if cfg.use_prediction_module:
            a = src.block(x)
            b = src.conv(x)
            b = src.bn(b)
            b = F.relu(b)
            x = a + b
        bbox_x = src.bbox_extra(x)
        conf_x = src.conf_extra(x)
        mask_x = src.mask_extra(x)
        bbox = src.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = src.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        if cfg.eval_mask_branch:
            mask = src.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
        else:
            mask = torch.zeros(x.size(0), bbox.size(1), self.mask_dim, device=bbox.device)
        if cfg.use_instance_coeff:
            inst = src.inst_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, cfg.num_instance_coeffs)
        if cfg.use_yolo_regressors:
            bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h
        if cfg.eval_mask_branch:
            if cfg.mask_type == mask_type.direct:
                mask = torch.sigmoid(mask)
            elif cfg.mask_type == mask_type.lincomb:
                mask = cfg.mask_proto_coeff_activation(mask)
                if cfg.mask_proto_coeff_gate:
                    gate = src.gate_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
                    mask = mask * torch.sigmoid(gate)
        priors = self.make_priors(conv_h, conv_w)
        preds = {'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors}
        if cfg.use_instance_coeff:
            preds['inst'] = inst
        return preds

    def make_priors(self, conv_h, conv_w):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        with timer.env('makepriors'):
            if self.last_conv_size != (conv_w, conv_h):
                prior_data = []
                for j, i in product(range(conv_h), range(conv_w)):
                    x = (i + 0.5) / conv_w
                    y = (j + 0.5) / conv_h
                    for scale, ars in zip(self.scales, self.aspect_ratios):
                        for ar in ars:
                            if not cfg.backbone.preapply_sqrt:
                                ar = sqrt(ar)
                            if cfg.backbone.use_pixel_scales:
                                if type(cfg.max_size) == tuple:
                                    width, height = cfg.max_size
                                    w = scale * ar / width
                                    h = scale / ar / height
                                else:
                                    w = scale * ar / cfg.max_size
                                    h = scale / ar / cfg.max_size
                            else:
                                w = scale * ar / conv_w
                                h = scale / ar / conv_h
                            if cfg.backbone.use_square_anchors:
                                h = w
                            prior_data += [x, y, w, h]
                self.priors = torch.Tensor(prior_data).view(-1, 4)
                self.last_conv_size = conv_w, conv_h
        return self.priors


class PredictionModuleTRT(PredictionModule):

    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None, index=0):
        super().__init__(in_channels, out_channels, aspect_ratios, scales, parent, index)
        if cfg.mask_proto_coeff_activation == torch.tanh:
            self.activation_func = nn.Tanh()
        else:
            raise NotImplementedError

    def forward(self, x):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        conv_h = x.size(2)
        conv_w = x.size(3)
        if cfg.extra_head_net is not None:
            x = self.upfeature(x)
        if cfg.use_prediction_module:
            a = self.block(x)
            b = self.conv(x)
            b = self.bn(b)
            b = F.relu(b)
            x = a + b
        bbox_x = self.bbox_extra(x)
        conf_x = self.conf_extra(x)
        mask_x = self.mask_extra(x)
        bbox = self.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = self.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        if cfg.eval_mask_branch:
            mask = self.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
        else:
            mask = torch.zeros(x.size(0), bbox.size(1), self.mask_dim, device=bbox.device)
        if cfg.use_instance_coeff:
            inst = self.inst_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, cfg.num_instance_coeffs)
            raise NotImplementedError
        if cfg.use_yolo_regressors:
            bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h
        if cfg.eval_mask_branch:
            if cfg.mask_type == mask_type.direct:
                mask = torch.sigmoid(mask)
            elif cfg.mask_type == mask_type.lincomb:
                mask = self.activation_func(mask)
                if cfg.mask_proto_coeff_gate:
                    gate = self.gate_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
                    mask = mask * torch.sigmoid(gate)
        return bbox, conf, mask


class Cat(nn.Module):

    def forward(self, a, b):
        x = torch.cat((a, b), dim=1)
        return x


class ShuffleCat(nn.Module):

    def forward(self, a, b):
        assert a.size() == b.size()
        n, c, h, w = a.size()
        a = a.permute(0, 2, 3, 1).contiguous().view(-1, c)
        b = b.permute(0, 2, 3, 1).contiguous().view(-1, c)
        x = torch.cat((a, b), dim=0).transpose(1, 0).contiguous()
        x = x.view(c * 2, n, h, w).permute(1, 0, 2, 3)
        return x


class ShuffleCatChunk(nn.Module):

    def forward(self, a, b):
        assert a.size() == b.size()
        n, c, h, w = a.size()
        a = torch.chunk(a, chunks=c, dim=1)
        b = torch.chunk(b, chunks=c, dim=1)
        x = [None] * (c * 2)
        x[::2] = a
        x[1::2] = b
        x = torch.cat(x, dim=1)
        return x


class ShuffleCatAlt(nn.Module):

    def forward(self, a, b):
        assert a.size() == b.size()
        n, c, h, w = a.size()
        x = torch.zeros(n, c * 2, h, w, dtype=a.dtype, device=a.device)
        x[:, ::2] = a
        x[:, 1::2] = b
        return x


class FlowNetUnwrap(nn.Module):

    def forward(self, preds):
        outs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        flow1, scale1, bias1, flow2, scale2, bias2, flow3, scale3, bias3 = preds
        outs.append((flow1, scale1, bias1))
        outs.append((flow2, scale2, bias2))
        outs.append((flow3, scale3, bias3))
        return outs


class FlowNetMiniTRTWrapper(nn.Module):

    def __init__(self, flow_net):
        super().__init__()
        self.flow_net = flow_net
        if cfg.flow.use_shuffle_cat:
            self.cat = ShuffleCat()
        else:
            self.cat = Cat()
        self.unwrap = FlowNetUnwrap()

    def forward(self, a, b):
        concat = self.cat(a, b)
        dummy_tensor = torch.tensor(0, dtype=a.dtype)
        preds = [dummy_tensor, dummy_tensor, dummy_tensor]
        preds_ = self.flow_net(concat)
        preds.extend(preds_)
        outs = self.unwrap(preds)
        return outs


class PredictionModuleTRTWrapper(nn.Module):

    def __init__(self, pred_layer):
        super().__init__()
        self.pred_layer = PredictionModuleTRT(*pred_layer.params[:-2], None, pred_layer.params[-1])
        pred_layer_w = pred_layer.parent[0] if pred_layer.parent[0] is not None else pred_layer
        self.pred_layer.load_state_dict(pred_layer_w.state_dict())

    def to_tensorrt(self, int8_mode=False, calibration_dataset=None, batch_size=1):
        if int8_mode:
            trt_fn = partial(torch2trt, int8_mode=True, int8_calib_dataset=calibration_dataset, strict_type_constraints=True, max_batch_size=batch_size)
        else:
            trt_fn = partial(torch2trt, fp16_mode=True, strict_type_constraints=True, max_batch_size=batch_size)
        input_sizes = [(1, 256, 69, 69), (1, 256, 35, 35), (1, 256, 18, 18), (1, 256, 9, 9), (1, 256, 5, 5)]
        x = torch.ones(input_sizes[self.pred_layer.index])
        self.pred_layer_torch = self.pred_layer
        self.pred_layer = trt_fn(self.pred_layer, [x])

    def forward(self, x):
        conv_h = x.size(2)
        conv_w = x.size(3)
        bbox, conf, mask = self.pred_layer(x)
        priors = self.pred_layer_torch.make_priors(conv_h, conv_w)
        preds = {'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors}
        return preds


class NoReLUBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(NoReLUBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class FlowNetMiniPredLayer(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, 2, kernel_size=3, padding=1, bias=False)
        self.scale = nn.Conv2d(in_features, cfg.fpn.num_features, kernel_size=1, padding=0, bias=True)
        self.bias = nn.Conv2d(in_features, cfg.fpn.num_features, kernel_size=1, padding=0, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
        nn.init.constant_(self.scale.bias, 1)

    def forward(self, x):
        offset = self.conv(x)
        scale = self.scale(x)
        bias = self.bias(x)
        return offset, scale, bias


class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.use_cross_class_nms = False
        self.use_fast_nms = False

    def __call__(self, predictions):
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            mask_data: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_data: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]
        
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """
        loc_data = predictions['loc']
        conf_data = predictions['conf']
        mask_data = predictions['mask']
        prior_data = predictions['priors']
        proto_data = predictions['proto'] if 'proto' in predictions else None
        inst_data = predictions['inst'] if 'inst' in predictions else None
        out = []
        with timer.env('Detect'):
            batch_size = loc_data.size(0)
            num_priors = prior_data.size(0)
            conf_preds = conf_data.view(batch_size, num_priors, self.num_classes).transpose(2, 1).contiguous()
            for batch_idx in range(batch_size):
                decoded_boxes = decode(loc_data[batch_idx], prior_data)
                result = self.detect(batch_idx, conf_preds, decoded_boxes, mask_data, inst_data)
                if result is not None and proto_data is not None:
                    result['proto'] = proto_data[batch_idx]
                out.append(result)
        return out

    def detect(self, batch_idx, conf_preds, decoded_boxes, mask_data, inst_data):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        cur_scores = conf_preds[batch_idx, 1:, :]
        conf_scores, _ = torch.max(cur_scores, dim=0)
        keep = conf_scores > self.conf_thresh
        scores = cur_scores[:, keep]
        boxes = decoded_boxes[keep, :]
        masks = mask_data[batch_idx, keep, :]
        if inst_data is not None:
            inst = inst_data[batch_idx, keep, :]
        if scores.size(1) == 0:
            return None
        if self.use_fast_nms:
            if self.use_cross_class_nms:
                boxes, masks, classes, scores = self.cc_fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
            else:
                boxes, masks, classes, scores = self.fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
        else:
            boxes, masks, classes, scores = self.traditional_nms(boxes, masks, scores, self.nms_thresh, self.conf_thresh)
        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}

    def coefficient_nms(self, coeffs, scores, cos_threshold=0.9, top_k=400):
        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]
        coeffs_norm = F.normalize(coeffs[idx], dim=1)
        cos_similarity = coeffs_norm @ coeffs_norm.t()
        cos_similarity.triu_(diagonal=1)
        cos_max, _ = torch.max(cos_similarity, dim=0)
        idx_out = idx[cos_max <= cos_threshold]
        return idx_out, idx_out.size(0)

    def cc_fast_nms(self, boxes, masks, scores, iou_threshold: float=0.5, top_k: int=200):
        scores, classes = scores.max(dim=0)
        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]
        boxes_idx = torch.index_select(boxes, 0, idx)
        iou = jaccard(boxes_idx, boxes_idx)
        iou.triu_(diagonal=1)
        iou_max, _ = torch.max(iou, dim=0)
        idx_keep = torch.nonzero(iou_max <= iou_threshold, as_tuple=True)[0]
        idx_out = torch.index_select(idx, 0, idx_keep)
        return tuple([torch.index_select(x, 0, idx_out) for x in (boxes, masks, classes, scores)])

    def fast_nms(self, boxes, masks, scores, iou_threshold: float=0.5, top_k: int=200, second_threshold: bool=False):
        scores, idx = scores.sort(1, descending=True)
        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]
        num_classes, num_dets = idx.size()
        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)
        iou = jaccard(boxes, boxes)
        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)
        keep = iou_max <= iou_threshold
        if second_threshold:
            keep *= scores > self.conf_thresh
        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)

        def fix_shape(classes, boxes, masks, scores):
            num_dets = torch.numel(classes)
            classes = classes.view(num_dets)
            boxes = boxes.view(num_dets, 4)
            masks = masks.view(num_dets, -1)
            scores = scores.view(num_dets)
            return classes, boxes, masks, scores

        def flatten_index_select(x, idx, end_dim=None):
            x = torch.flatten(x, end_dim=end_dim)
            return torch.index_select(x, 0, idx)
        if not cfg.use_tensorrt_safe_mode:
            classes = classes[keep]
            boxes = boxes[keep]
            masks = masks[keep]
            scores = scores[keep]
        else:
            keep = torch.flatten(keep, end_dim=1)
            idx = torch.nonzero(keep, as_tuple=True)[0]
            classes, boxes, masks, scores = [flatten_index_select(x, idx, end_dim=1) for x in (classes, boxes, masks, scores)]
            classes, boxes, masks, scores = fix_shape(classes, boxes, masks, scores)
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]
        if not cfg.use_tensorrt_safe_mode:
            classes = classes[idx]
            boxes = boxes[idx]
            masks = masks[idx]
        else:
            classes = torch.index_select(classes, 0, idx)
            boxes = torch.index_select(boxes, 0, idx)
            masks = torch.index_select(masks, 0, idx)
            classes, boxes, masks, scores = fix_shape(classes, boxes, masks, scores)
        return boxes, masks, classes, scores

    def traditional_nms(self, boxes, masks, scores, iou_threshold=0.5, conf_thresh=0.05):
        num_classes = scores.size(0)
        idx_lst = []
        cls_lst = []
        scr_lst = []
        boxes = boxes * cfg.max_size
        for _cls in range(num_classes):
            cls_scores = scores[_cls, :]
            conf_mask = cls_scores > conf_thresh
            idx = torch.arange(cls_scores.size(0), device=boxes.device)
            cls_scores = cls_scores[conf_mask]
            idx = idx[conf_mask]
            if cls_scores.size(0) == 0:
                continue
            preds = torch.cat([boxes[conf_mask], cls_scores[:, None]], dim=1).cpu().numpy()
            keep = cnms(preds, iou_threshold)
            keep = torch.Tensor(keep, device=boxes.device).long()
            idx_lst.append(idx[keep])
            cls_lst.append(keep * 0 + _cls)
            scr_lst.append(cls_scores[keep])
        idx = torch.cat(idx_lst, dim=0)
        classes = torch.cat(cls_lst, dim=0)
        scores = torch.cat(scr_lst, dim=0)
        scores, idx2 = scores.sort(0, descending=True)
        idx2 = idx2[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]
        idx = idx[idx2]
        classes = classes[idx2]
        return boxes[idx] / cfg.max_size, masks[idx], classes, scores


def conv_lrelu(in_features, out_features, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=dilation, dilation=dilation, groups=groups), nn.LeakyReLU(0.1, inplace=True))


def build_flow_convs(encode_layers, in_features, out_features, stride=1, groups=1):
    conv = []
    conv.append(conv_lrelu(in_features, cfg.flow.encode_channels * encode_layers[0], groups=groups, stride=stride))
    for encode_idx, encode_layer in enumerate(encode_layers[:-1]):
        conv.append(conv_lrelu(cfg.flow.encode_channels * encode_layers[encode_idx], cfg.flow.encode_channels * encode_layers[encode_idx + 1], groups=groups))
    conv.append(conv_lrelu(cfg.flow.encode_channels * encode_layers[-1], out_features))
    return nn.Sequential(*conv)


def shuffle_cat(a, b):
    assert a.size() == b.size()
    n, c, h, w = a.size()
    a = a.permute(0, 2, 3, 1).contiguous().view(-1, c)
    b = b.permute(0, 2, 3, 1).contiguous().view(-1, c)
    x = torch.cat((a, b), dim=0).transpose(1, 0).contiguous()
    x = x.view(c * 2, n, h, w).permute(1, 0, 2, 3)
    return x


def construct_backbone(cfg):
    """ Constructs a backbone given a backbone config object (see config.py). """
    backbone = cfg.type(*cfg.args)
    num_layers = max(cfg.selected_layers) + 1
    while len(backbone.layers) < num_layers:
        backbone.add_layer()
    return backbone


cache = {}


@torch.jit.ignore
def generate_grid_as(n: int, h: int, w: int, t: torch.Tensor):
    if (n, h, w) in cache:
        return cache[n, h, w].clone()
    x_ = torch.arange(w, dtype=t.dtype, device=t.device).view(1, -1).expand(h, -1)
    y_ = torch.arange(h, dtype=t.dtype, device=t.device).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float()
    grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
    cache[n, h, w] = grid.clone()
    return grid.clone()


class Yolact(nn.Module):
    """


    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 


    You can set the arguments by chainging them in the backbone config object in config.py.

    Parameters (in cfg.backbone):
        - selected_layers: The indices of the conv layers to use for prediction.
        - pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
        - pred_aspect_ratios: A list of lists of aspect ratios with len(selected_layers) (see PredictionModule)
    """

    def __init__(self, training=True):
        super().__init__()
        self.backbone = construct_backbone(cfg.backbone)
        self.training = training
        if cfg.freeze_bn:
            self.freeze_bn()
        if cfg.mask_type == mask_type.direct:
            cfg.mask_dim = cfg.mask_size ** 2
        elif cfg.mask_type == mask_type.lincomb:
            if cfg.mask_proto_use_grid:
                self.grid = torch.Tensor(np.load(cfg.mask_proto_grid_file))
                self.num_grids = self.grid.size(0)
            else:
                self.num_grids = 0
            self.proto_src = cfg.mask_proto_src
            if self.proto_src is None:
                in_channels = 3
            elif cfg.fpn is not None:
                in_channels = cfg.fpn.num_features
            else:
                in_channels = self.backbone.channels[self.proto_src]
            in_channels += self.num_grids
            self.proto_net, cfg.mask_dim = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False)
            if cfg.mask_proto_bias:
                cfg.mask_dim += 1
        self.selected_layers = cfg.backbone.selected_layers
        src_channels = self.backbone.channels
        if cfg.fpn is not None:
            if cfg.flow is not None:
                self.fpn_phase_1 = FPN_phase_1([src_channels[i] for i in self.selected_layers])
                self.fpn_phase_2 = FPN_phase_2([src_channels[i] for i in self.selected_layers])
                if cfg.flow.use_spa:
                    self.spa = SPA(len(self.selected_layers))
                if cfg.flow.warp_mode == 'flow':
                    if cfg.flow.model == 'mini':
                        lateral_channels = cfg.fpn.num_features
                        if len(cfg.flow.reduce_channels) > 0:
                            lateral_channels = cfg.flow.reduce_channels[-1]
                        self.flow_net_pre_convs = FlowNetMiniPreConvs(cfg.flow.reduce_channels)
                        if not training and (cfg.torch2trt_flow_net or cfg.torch2trt_flow_net_int8):
                            self.flow_net = FlowNetMiniTRT(lateral_channels * 2)
                        else:
                            self.flow_net = FlowNetMini(lateral_channels * 2)
                    else:
                        raise NotImplementedError
                self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))
            else:
                self.fpn = FPN([src_channels[i] for i in self.selected_layers])
                self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))
            src_channels = [cfg.fpn.num_features] * len(self.selected_layers)
        self.prediction_layers = nn.ModuleList()
        for idx, layer_idx in enumerate(self.selected_layers):
            parent = None
            if cfg.share_prediction_module and idx > 0:
                parent = self.prediction_layers[0]
            pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx], aspect_ratios=cfg.backbone.pred_aspect_ratios[idx], scales=cfg.backbone.pred_scales[idx], parent=parent, index=idx)
            self.prediction_layers.append(pred)
        if cfg.use_class_existence_loss:
            self.class_existence_fc = nn.Linear(src_channels[-1], cfg.num_classes - 1)
        if cfg.use_semantic_segmentation_loss:
            self.semantic_seg_conv = nn.Conv2d(src_channels[0], cfg.num_classes - 1, kernel_size=1)
        self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=200, conf_thresh=0.05, nms_thresh=0.5)

    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)

    def load_weights(self, path, args=None):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path, map_location='cpu')
        cur_state_dict = self.state_dict()
        if args is not None and args.drop_weights is not None:
            drop_weight_keys = args.drop_weights.split(',')
        transfered_from_yolact = False
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]
            if key.startswith('fpn.downsample_layers.'):
                if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
                    del state_dict[key]
            if args is not None and args.drop_weights is not None:
                for drop_key in drop_weight_keys:
                    if key.startswith(drop_key):
                        del state_dict[key]
            if key.startswith('fpn.lat_layers'):
                transfered_from_yolact = True
                state_dict[key.replace('fpn.', 'fpn_phase_1.')] = state_dict[key]
                del state_dict[key]
            elif key.startswith('fpn.') and key in state_dict:
                transfered_from_yolact = True
                state_dict[key.replace('fpn.', 'fpn_phase_2.')] = state_dict[key]
                del state_dict[key]
        keys_not_exist = []
        keys_not_used = []
        keys_mismatch = []
        for key in list(cur_state_dict.keys()):
            if args is not None and args.drop_weights is not None:
                for drop_key in drop_weight_keys:
                    if key.startswith(drop_key):
                        state_dict[key] = cur_state_dict[key]
            if key not in state_dict:
                keys_not_exist.append(key)
                state_dict[key] = cur_state_dict[key]
            elif state_dict[key].size() != cur_state_dict[key].size():
                keys_mismatch.append(key)
                state_dict[key] = cur_state_dict[key]
        for key in list(state_dict.keys()):
            if key not in cur_state_dict:
                keys_not_used.append(key)
                del state_dict[key]
        logger = logging.getLogger('yolact.model.load')
        if len(keys_not_used) > 0:
            logger.warning('Some parameters in the checkpoint are not used: {}'.format(', '.join(keys_not_used)))
        if len(keys_not_exist) > 0:
            logger.warning('Some parameters required by the model do not exist in the checkpoint, and are initialized as they should be: {}'.format(', '.join(keys_not_exist)))
        if len(keys_mismatch) > 0:
            logger.warning('Some parameters in the checkpoint have a different shape in the current model, and are initialized as they should be: {}'.format(', '.join(keys_mismatch)))
        if args is not None and (args.coco_transfer or args.yolact_transfer):
            logger.warning('`--coco_transfer` or `--yolact_transfer` is no longer needed. The code will automatically detect and convert YOLACT-trained weights now.')
        self.load_state_dict(state_dict)
        if not self.training:
            self.create_partial_backbone()
            if cfg.torch2trt_flow_net or cfg.torch2trt_flow_net_int8:
                self.create_embed_flow_net()

    def init_weights(self, backbone_path):
        """ Initialize weights for training. """
        self.backbone.init_backbone(backbone_path)
        conv_constants = getattr(nn.Conv2d(1, 1, 1), '__constants__')

        def all_in(x, y):
            for _x in x:
                if _x not in y:
                    return False
            return True
        for name, module in self.named_modules():
            is_script_conv = False
            if 'Script' in type(module).__name__:
                if hasattr(module, 'original_name'):
                    is_script_conv = 'Conv' in module.original_name
                else:
                    is_script_conv = all_in(module.__dict__['_constants_set'], conv_constants) and all_in(conv_constants, module.__dict__['_constants_set'])
            is_conv_layer = isinstance(module, nn.Conv2d) or is_script_conv
            if is_conv_layer and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    if cfg.use_focal_loss and 'conf_layer' in name:
                        if not cfg.use_sigmoid_focal_loss:
                            module.bias.data[0] = np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                            module.bias.data[1:] = -np.log(module.bias.size(0) - 1)
                        else:
                            module.bias.data[0] = -np.log(cfg.focal_loss_init_pi / (1 - cfg.focal_loss_init_pi))
                            module.bias.data[1:] = -np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                    else:
                        module.bias.data.zero_()

    def train(self, mode=True):
        super().train(mode)
        if cfg.freeze_bn:
            self.freeze_bn()
        if not mode:
            return
        if cfg.flow is not None and cfg.flow.fine_tune_layers is not None:
            self.fine_tune_layers()

    def freeze_bn(self):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad = False

    def fine_tune_layers(self):
        fine_tune_layers = cfg.flow.fine_tune_layers
        freeze_or_ft = fine_tune_layers[0] == '-'
        if freeze_or_ft:
            fine_tune_layers = fine_tune_layers[1:]
        fine_tune_layer_names = fine_tune_layers.split(',')
        logger = logging.getLogger('yolact.train')
        freeze_layers = []
        fine_tune_layers = []
        for name, module in self.named_children():
            name_in_list = name in fine_tune_layer_names
            if name_in_list == freeze_or_ft:
                freeze_layers.append(name)
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            else:
                fine_tune_layers.append(name)
        logger.info('Fine tuning weights of modules: {}'.format(', '.join(fine_tune_layers)))
        logger.info('Freezing weights of modules: {}'.format(', '.join(freeze_layers)))

    def extra_loss(self, net_outs, gt_net_outs):
        losses = {}
        if cfg.flow.feature_matching_loss is not None:

            def t(fea, layer_idx):
                fpn_net = self.fpn_phase_2
                pred_layer = fpn_net.pred_layers[layer_idx + 1]
                bias = pred_layer.bias.detach() if pred_layer.bias is not None else None
                fea = F.relu(F.conv2d(fea, weight=pred_layer.weight.detach(), bias=bias, stride=pred_layer.stride, padding=pred_layer.padding))
                return fea
            assert cfg.flow.fm_loss_loc in ('L', 'P', 'L+P')
            loss_W = 0
            pairs = []
            gt_outs_fpn = gt_net_outs['outs_phase_1'][1:]
            preds_outs_fpn = [net_outs['outs_phase_1'][1:]]
            if net_outs.get('direct_transform', None):
                preds_outs_fpn.append(net_outs['direct_transform'])
            for pred_outs_fpn in preds_outs_fpn:
                for layer_idx in range(2):
                    FPN_GTs = gt_outs_fpn[layer_idx]
                    FPN_preds = pred_outs_fpn[layer_idx]
                    if cfg.flow.fm_loss_loc != 'P':
                        pairs.append((FPN_GTs, FPN_preds))
                    if cfg.flow.fm_loss_loc != 'L':
                        pairs.append((t(FPN_GTs, layer_idx), t(FPN_preds, layer_idx)))
            for FPN_GTs, FPN_preds in pairs:
                n_, c_ = FPN_preds.size()[:2]
                if cfg.flow.feature_matching_loss == 'SmoothL1':
                    level_loss = F.smooth_l1_loss(FPN_preds, FPN_GTs, reduction='sum')
                    level_loss = level_loss / n_ / c_
                elif cfg.flow.feature_matching_loss == 'cosine':
                    level_loss = F.cosine_similarity(FPN_preds, FPN_GTs)
                    level_loss = (1 - level_loss).mean()
                else:
                    raise NotImplementedError
                loss_W += level_loss
            loss_W /= len(pairs)
            losses['W'] = loss_W * cfg.flow.fm_loss_alpha
        return losses

    def forward_flow(self, extras):
        imgs_1, imgs_2 = extras
        if cfg.flow.model == 'mini':
            feas_1 = self.backbone(imgs_1, partial=True)
            feas_2 = self.backbone(imgs_2, partial=True)
            fea_1 = feas_1[-1].detach()
            fea_2 = feas_2[-1].detach()
            src_lat_layer = self.fpn_phase_1.lat_layers[-1]
            src_lat_1 = src_lat_layer(fea_1).detach()
            src_lat_2 = src_lat_layer(fea_2).detach()
            src_lat_1 = self.flow_net_pre_convs(src_lat_1)
            src_lat_2 = self.flow_net_pre_convs(src_lat_2)
            preds_flow = self.flow_net(src_lat_1, src_lat_2)
            preds_flow = [pred[0] for pred in preds_flow]
        else:
            raise NotImplementedError
        return preds_flow

    def create_embed_flow_net(self):
        if hasattr(self, 'flow_net'):
            self.flow_net = FlowNetMiniTRTWrapper(self.flow_net)

    def create_partial_backbone(self):
        if cfg.flow.warp_mode == 'none':
            return
        logger = logging.getLogger('yolact.model.load')
        logger.debug('Creating partial backbone...')
        backbone = construct_backbone(cfg.backbone)
        backbone.load_state_dict(self.backbone.state_dict())
        backbone.layers = backbone.layers[:2]
        self.partial_backbone = backbone
        logger.debug('Partial backbone created...')

    def _get_trt_cache_path(self, module_name, int8_mode=False, batch_size=1):
        return '{}.{}{}{}.trt'.format(self.model_path, module_name, '.int8_{}'.format(cfg.torch2trt_max_calibration_images) if int8_mode else '', '_bs_{}'.format(batch_size))

    def has_trt_cached_module(self, module_name, int8_mode=False, batch_size=1):
        module_path = self._get_trt_cache_path(module_name, int8_mode, batch_size)
        return os.path.isfile(module_path)

    def load_trt_cached_module(self, module_name, int8_mode=False, batch_size=1):
        module_path = self._get_trt_cache_path(module_name, int8_mode, batch_size)
        if not os.path.isfile(module_path):
            return None
        module = TRTModule()
        module.load_state_dict(torch.load(module_path))
        return module

    def save_trt_cached_module(self, module, module_name, int8_mode=False, batch_size=1):
        module_path = self._get_trt_cache_path(module_name, int8_mode, batch_size)
        torch.save(module.state_dict(), module_path)

    def trt_load_if(self, module_name, trt_fn, trt_fn_params, int8_mode=False, parent=None, batch_size=1):
        if parent is None:
            parent = self
        if not hasattr(parent, module_name):
            return
        module = getattr(parent, module_name)
        trt_cache = self.load_trt_cached_module(module_name, int8_mode, batch_size=batch_size)
        if trt_cache is None:
            module = trt_fn(module, trt_fn_params)
            self.save_trt_cached_module(module, module_name, int8_mode, batch_size=batch_size)
        else:
            module = trt_cache
        setattr(parent, module_name, module)

    def to_tensorrt_backbone(self, int8_mode=False, calibration_dataset=None, batch_size=1):
        """Converts the Backbone to a TRTModule.
        """
        if int8_mode:
            trt_fn = partial(torch2trt, int8_mode=True, int8_calib_dataset=calibration_dataset, strict_type_constraints=True, max_batch_size=batch_size)
        else:
            trt_fn = partial(torch2trt, fp16_mode=True, strict_type_constraints=True, max_batch_size=batch_size)
        x = torch.ones((1, 3, cfg.max_size, cfg.max_size))
        self.trt_load_if('backbone', trt_fn, [x], int8_mode, batch_size=batch_size)
        self.trt_load_if('partial_backbone', trt_fn, [x], int8_mode, batch_size=batch_size)

    def to_tensorrt_protonet(self, int8_mode=False, calibration_dataset=None, batch_size=1):
        """Converts ProtoNet to a TRTModule.
        """
        if int8_mode:
            trt_fn = partial(torch2trt, int8_mode=True, int8_calib_dataset=calibration_dataset, strict_type_constraints=True, max_batch_size=batch_size)
        else:
            trt_fn = partial(torch2trt, fp16_mode=True, strict_type_constraints=True, max_batch_size=batch_size)
        x = torch.ones((1, 256, 69, 69))
        self.trt_load_if('proto_net', trt_fn, [x], int8_mode, batch_size=batch_size)

    def to_tensorrt_fpn(self, int8_mode=False, calibration_dataset=None, batch_size=1):
        """Converts FPN to a TRTModule.
        """
        if int8_mode:
            trt_fn = partial(torch2trt, int8_mode=True, int8_calib_dataset=calibration_dataset, strict_type_constraints=True, max_batch_size=batch_size)
        else:
            trt_fn = partial(torch2trt, fp16_mode=True, strict_type_constraints=True, max_batch_size=batch_size)
        self.lat_layer = self.fpn_phase_1.lat_layers[-1]
        if cfg.backbone.name == 'ResNet50' or cfg.backbone.name == 'ResNet101':
            x = [torch.randn(1, 512, 69, 69), torch.randn(1, 1024, 35, 35), torch.randn(1, 2048, 18, 18)]
        elif cfg.backbone.name == 'MobileNetV2':
            x = [torch.randn(1, 32, 69, 69), torch.randn(1, 64, 35, 35), torch.randn(1, 160, 18, 18)]
        else:
            raise ValueError('Backbone: {} is not currently supported with TensorRT.'.format(cfg.backbone.name))
        self.trt_load_if('fpn_phase_1', trt_fn, x, int8_mode, batch_size=batch_size)
        if cfg.backbone.name == 'ResNet50' or cfg.backbone.name == 'ResNet101':
            x = [torch.randn(1, 256, 69, 69), torch.randn(1, 256, 35, 35), torch.randn(1, 256, 18, 18)]
        elif cfg.backbone.name == 'MobileNetV2':
            x = [torch.randn(1, 256, 69, 69), torch.randn(1, 256, 35, 35), torch.randn(1, 256, 18, 18)]
        else:
            raise ValueError('Backbone: {} is not currently supported with TensorRT.'.format(cfg.backbone.name))
        self.trt_load_if('fpn_phase_2', trt_fn, x, int8_mode, batch_size=batch_size)
        trt_fn = partial(torch2trt, fp16_mode=True, strict_type_constraints=True)
        if cfg.backbone.name == 'ResNet50' or cfg.backbone.name == 'ResNet101':
            x = torch.randn(1, 512, 69, 69)
        elif cfg.backbone.name == 'MobileNetV2':
            x = torch.randn(1, 32, 69, 69)
        else:
            raise ValueError('Backbone: {} is not currently supported with TensorRT.'.format(cfg.backbone.name))
        self.trt_load_if('lat_layer', trt_fn, [x], int8_mode=False, batch_size=batch_size)

    def to_tensorrt_prediction_head(self, int8_mode=False, calibration_dataset=None, batch_size=1):
        """Converts Prediction Head to a TRTModule.
        """
        if int8_mode:
            trt_fn = partial(torch2trt, int8_mode=True, int8_calib_dataset=calibration_dataset, strict_type_constraints=True, max_batch_size=batch_size)
        else:
            trt_fn = partial(torch2trt, fp16_mode=True, strict_type_constraints=True, max_batch_size=batch_size)
        for idx, pred_layer in enumerate(self.prediction_layers):
            pred_layer = PredictionModuleTRTWrapper(pred_layer)
            pred_layer.to_tensorrt(batch_size=batch_size)
            self.prediction_layers[idx] = pred_layer

    def to_tensorrt_spa(self, int8_mode=False, calibration_dataset=None, batch_size=1):
        """Converts SPA to a TRTModule.
        """
        if int8_mode:
            trt_fn = partial(torch2trt, int8_mode=True, int8_calib_dataset=calibration_dataset, strict_type_constraints=True, max_batch_size=batch_size)
        else:
            trt_fn = partial(torch2trt, fp16_mode=True, strict_type_constraints=True, max_batch_size=batch_size)
        c3 = torch.ones((1, 256, 69, 69))
        f2 = torch.ones((1, 256, 35, 35))
        f3 = torch.ones((1, 256, 18, 18))
        self.trt_load_if('spa', trt_fn, [c3, f2, f3], int8_mode, parent=self.spa, batch_size=batch_size)

    def to_tensorrt_flow_net(self, int8_mode=False, calibration_dataset=None, batch_size=1):
        """Converts FlowNet to a TRTModule.
        """
        if int8_mode:
            trt_fn = partial(torch2trt, int8_mode=True, int8_calib_dataset=calibration_dataset, strict_type_constraints=True, max_batch_size=batch_size)
        else:
            trt_fn = partial(torch2trt, fp16_mode=True, strict_type_constraints=True, max_batch_size=batch_size)
        lateral_channels = cfg.fpn.num_features
        if len(cfg.flow.reduce_channels) > 0:
            lateral_channels = cfg.flow.reduce_channels[-1]
        x = torch.ones((1, lateral_channels * 2, 69, 69))
        self.trt_load_if('flow_net', trt_fn, [x], int8_mode, parent=self.flow_net, batch_size=batch_size)

    def forward(self, x, extras=None):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        if cfg.flow.train_flow:
            return self.forward_flow(extras)
        outs_wrapper = {}
        with timer.env('backbone'):
            if cfg.flow is None or extras is None or extras['backbone'] == 'full':
                outs = self.backbone(x)
            elif extras is not None and extras['backbone'] == 'partial':
                if hasattr(self, 'partial_backbone'):
                    outs = self.partial_backbone(x)
                else:
                    outs = self.backbone(x, partial=True)
            else:
                raise NotImplementedError
        if cfg.flow is not None:
            with timer.env('fpn'):
                assert type(extras) == dict
                if extras['backbone'] == 'full':
                    outs = [outs[i] for i in cfg.backbone.selected_layers]
                    outs_fpn_phase_1_wrapper = self.fpn_phase_1(*outs)
                    outs_phase_1, lats_phase_1 = outs_fpn_phase_1_wrapper[:len(outs)], outs_fpn_phase_1_wrapper[len(outs):]
                    lateral = lats_phase_1[0].detach()
                    moving_statistics = extras['moving_statistics']
                    if extras.get('keep_statistics', False):
                        outs_wrapper['feats'] = [out.detach() for out in outs_phase_1]
                        outs_wrapper['lateral'] = lateral
                    outs_wrapper['outs_phase_1'] = [out.detach() for out in outs_phase_1]
                else:
                    assert extras['moving_statistics'] is not None
                    moving_statistics = extras['moving_statistics']
                    outs_phase_1 = moving_statistics['feats'].copy()
                    if cfg.flow.warp_mode != 'take':
                        src_conv = outs[-1].detach()
                        src_lat_layer = self.lat_layer if hasattr(self, 'lat_layer') else self.fpn_phase_1.lat_layers[-1]
                        lateral = src_lat_layer(src_conv).detach()
                    if cfg.flow.warp_mode == 'flow':
                        with timer.env('flow'):
                            flows = self.flow_net(self.flow_net_pre_convs(lateral), self.flow_net_pre_convs(moving_statistics['lateral']))
                            preds_feat = list()
                            if cfg.flow.flow_layer == 'top':
                                flows = [flows[0] for _ in flows]
                            if cfg.flow.warp_layers == 'P4P5':
                                flows = flows[1:]
                                outs_phase_1 = outs_phase_1[1:]
                            for (flow, scale_factor, scale_bias), feat in zip(flows, outs_phase_1):
                                if cfg.flow.flow_layer == 'top':
                                    _, _, h, w = feat.size()
                                    _, _, h_, w_ = flow.size()
                                    if (h, w) != (h_, w_):
                                        flow = F.interpolate(flow, size=(h, w), mode='area')
                                        scale_factor = F.interpolate(scale_factor, size=(h, w), mode='area')
                                        scale_bias = F.interpolate(scale_bias, size=(h, w), mode='area')
                                pred_feat = deform_op(feat, flow)
                                if cfg.flow.use_scale_factor:
                                    pred_feat *= scale_factor
                                if cfg.flow.use_scale_bias:
                                    pred_feat += scale_bias
                                preds_feat.append(pred_feat)
                            outs_wrapper['preds_flow'] = [[x.detach() for x in flow] for flow in flows]
                        outs_phase_1 = preds_feat
                    if cfg.flow.warp_layers == 'P4P5':
                        with timer.env('p3'):
                            _, _, h, w = src_conv.size()
                            src_fpn = outs_phase_1[0]
                            src_fpn = F.interpolate(src_fpn, size=(h, w), mode=cfg.fpn.interpolation_mode, align_corners=False)
                            p3 = src_fpn + lateral
                            outs_phase_1 = [p3] + outs_phase_1
                    if cfg.flow.use_spa:
                        with timer.env('spa'):
                            fpn_outs = outs_phase_1.copy()
                            outs_phase_1 = [fpn_outs[0]]
                            outs_ = self.spa(lateral, *fpn_outs[1:])
                            outs_phase_1.extend(outs_)
                    outs_wrapper['outs_phase_1'] = outs_phase_1.copy()
                outs = self.fpn_phase_2(*outs_phase_1)
                if extras['backbone'] == 'partial':
                    outs_wrapper['outs_phase_2'] = [out for out in outs]
                else:
                    outs_wrapper['outs_phase_2'] = [out.detach() for out in outs]
        elif cfg.fpn is not None:
            with timer.env('fpn'):
                outs = [outs[i] for i in cfg.backbone.selected_layers]
                outs = self.fpn(outs)
        if extras is not None and extras.get('interrupt', None):
            return outs_wrapper
        proto_out = None
        if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
            with timer.env('proto'):
                proto_x = x if self.proto_src is None else outs[self.proto_src]
                if self.num_grids > 0:
                    grids = self.grid.repeat(proto_x.size(0), 1, 1, 1)
                    proto_x = torch.cat([proto_x, grids], dim=1)
                proto_out = self.proto_net(proto_x)
                proto_out = cfg.mask_proto_prototype_activation(proto_out)
                if cfg.mask_proto_prototypes_as_features:
                    proto_downsampled = proto_out.clone()
                    if cfg.mask_proto_prototypes_as_features_no_grad:
                        proto_downsampled = proto_out.detach()
                proto_out = proto_out.permute(0, 2, 3, 1).contiguous()
                if cfg.mask_proto_bias:
                    bias_shape = [x for x in proto_out.size()]
                    bias_shape[-1] = 1
                    proto_out = torch.cat([proto_out, torch.ones(*bias_shape)], -1)
        with timer.env('pred_heads'):
            pred_outs = {'loc': [], 'conf': [], 'mask': [], 'priors': []}
            if cfg.use_instance_coeff:
                pred_outs['inst'] = []
            for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
                pred_x = outs[idx]
                if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_prototypes_as_features:
                    proto_downsampled = F.interpolate(proto_downsampled, size=outs[idx].size()[2:], mode='bilinear', align_corners=False)
                    pred_x = torch.cat([pred_x, proto_downsampled], dim=1)
                if self.training or not (cfg.torch2trt_prediction_module or cfg.torch2trt_prediction_module_int8):
                    if cfg.share_prediction_module and pred_layer is not self.prediction_layers[0]:
                        pred_layer.parent = [self.prediction_layers[0]]
                p = pred_layer(pred_x)
                for k, v in p.items():
                    pred_outs[k].append(v)
        for k, v in pred_outs.items():
            pred_outs[k] = torch.cat(v, -2)
        if proto_out is not None:
            pred_outs['proto'] = proto_out
        if self.training:
            if cfg.use_class_existence_loss:
                pred_outs['classes'] = self.class_existence_fc(outs[-1].mean(dim=(2, 3)))
            if cfg.use_semantic_segmentation_loss:
                pred_outs['segm'] = self.semantic_seg_conv(outs[0])
            outs_wrapper['pred_outs'] = pred_outs
        else:
            if cfg.use_sigmoid_focal_loss:
                pred_outs['conf'] = torch.sigmoid(pred_outs['conf'])
            elif cfg.use_objectness_score:
                objectness = torch.sigmoid(pred_outs['conf'][:, :, 0])
                pred_outs['conf'][:, :, 1:] = objectness[:, :, None] * F.softmax(pred_outs['conf'][:, :, 1:], -1)
                pred_outs['conf'][:, :, 0] = 1 - objectness
            else:
                pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)
            outs_wrapper['pred_outs'] = self.detect(pred_outs)
        return outs_wrapper


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Cat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBNAct,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DarkNetBackbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (FlowNetUnwrap,
     lambda: ([], {}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OpticalFlowLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResNetBackbone,
     lambda: ([], {'layers': [4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ResNetBackboneGN,
     lambda: ([], {'layers': [4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ShuffleCat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShuffleCatAlt,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShuffleCatChunk,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (VGGBackbone,
     lambda: ([], {'cfg': _mock_config()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_haotian_liu_yolact_edge(_paritybench_base):
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

