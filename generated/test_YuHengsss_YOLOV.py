import sys
_module = sys.modules[__name__]
del sys
build = _module
convert_weights = _module
demo = _module
dump = _module
models = _module
darknet = _module
network_blocks = _module
yolo_fpn = _module
yolo_head = _module
yolo_pafpn = _module
yolox = _module
onnx_inference = _module
openvino_inference = _module
conf = _module
default = _module
yolov3 = _module
yolox_l = _module
yolox_m = _module
yolox_nano = _module
yolox_s = _module
yolox_tiny = _module
yolox_x = _module
yolov_base = _module
yolov_l = _module
yolov_l_online = _module
yolov_s = _module
yolov_s_online = _module
yolov_x = _module
yolov_x_online = _module
yoloxl_vid = _module
yoloxs_vid = _module
yoloxx_vid = _module
yolovl_ovis_75_75_750 = _module
yolovs_ovis_75_75_750 = _module
yolovx_ovis_75_75_750 = _module
yoloxl_ovis = _module
yoloxs_ovis = _module
yoloxx_ovis = _module
setup = _module
REPP = _module
REPPM = _module
demo = _module
eval = _module
imagenet_vid_eval_motion = _module
motion_utils = _module
repp_utils = _module
train = _module
val_to_imdb = _module
val_to_imdb_online = _module
vid_demo = _module
vid_demo_wpost = _module
vid_eval = _module
vid_train = _module
yolov_demo_online = _module
core = _module
launch = _module
trainer = _module
vid_trainer = _module
data = _module
data_augment = _module
data_prefetcher = _module
dataloading = _module
datasets = _module
argoverse = _module
coco = _module
coco_classes = _module
datasets_wrapper = _module
mosaicdetection = _module
ovis = _module
vid = _module
vid_classes = _module
voc = _module
voc_classes = _module
samplers = _module
evaluators = _module
coco_evaluator = _module
vid_evaluator_v2 = _module
voc_eval = _module
voc_evaluator = _module
exp = _module
base_exp = _module
yolox_base = _module
layers = _module
fast_coco_eval_api = _module
jit_ops = _module
build = _module
darknet = _module
darknet53 = _module
initializer = _module
losses = _module
myolox = _module
network_blocks = _module
post_process = _module
post_trans = _module
weight_init = _module
yolo_fpn = _module
yolo_head = _module
yolo_pafpn = _module
yolov_msa_online = _module
yolov_online = _module
yolovp_msa = _module
yolox = _module
tools = _module
utils = _module
allreduce_norm = _module
box_op = _module
boxes = _module
checkpoint = _module
compat = _module
demo_utils = _module
dist = _module
ema = _module
logger = _module
lr_scheduler = _module
metric = _module
model_utils = _module
setup_env = _module
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


from collections import OrderedDict


import torch


from typing import Dict


from typing import List


from typing import Tuple


import torch.nn as nn


import random


import torch.distributed as dist


import re


import time


import warnings


import torch.backends.cudnn as cudnn


from torch.nn.parallel import DistributedDataParallel as DDP


import numpy as np


import copy


from random import shuffle


import torch.multiprocessing as mp


from torch.utils.tensorboard import SummaryWriter


import uuid


from torch.utils.data.dataloader import DataLoader as torchDataLoader


from torch.utils.data.dataloader import default_collate


from functools import wraps


from torch.utils.data.dataset import ConcatDataset as torchConcatDataset


from torch.utils.data.dataset import Dataset as torchDataset


import numpy


from torch.utils.data.sampler import Sampler


from torch.utils.data.sampler import BatchSampler


from torch.utils.data.sampler import SequentialSampler


import math


import itertools


from typing import Optional


from torch.utils.data.sampler import BatchSampler as torchBatchSampler


from collections import ChainMap


from abc import ABCMeta


from abc import abstractmethod


from torch.nn import Module


from torch import nn


from torch.hub import load_state_dict_from_url


import torchvision


from torch.nn import functional as F


from matplotlib import pyplot as plt


import torch.nn.functional as F


from scipy.optimize import linear_sum_assignment


from torchvision.ops import roi_align


from torch import distributed as dist


from torchvision.ops.boxes import box_area


import functools


from copy import deepcopy


import inspect


from collections import defaultdict


from collections import deque


from typing import Sequence


def get_activation(name='silu', inplace=True):
    if name == 'silu':
        module = nn.SiLU(inplace=inplace)
    elif name == 'relu':
        module = nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == 'swish':
        module = nn.SiLU(inplace=inplace)
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act='silu'):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ResLayer(nn.Module):
    """Residual layer with `in_channels` inputs."""

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(in_channels, mid_channels, ksize=1, stride=1, act='lrelu')
        self.layer2 = BaseConv(mid_channels, in_channels, ksize=3, stride=1, act='lrelu')

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation='silu'):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class Darknet(nn.Module):
    depth2blocks = {(21): [1, 2, 2, 1], (53): [2, 8, 8, 4]}

    def __init__(self, depth, in_channels=3, stem_out_channels=32, out_features=('dark3', 'dark4', 'dark5')):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output channels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, 'please provide output features of Darknet'
        self.out_features = out_features
        self.stem = nn.Sequential(BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act='lrelu'), *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2))
        in_channels = stem_out_channels * 2
        num_blocks = Darknet.depth2blocks[depth]
        self.dark2 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[0], stride=2))
        in_channels *= 2
        self.dark3 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[1], stride=2))
        in_channels *= 2
        self.dark4 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[2], stride=2))
        in_channels *= 2
        self.dark5 = nn.Sequential(*self.make_group_layer(in_channels, num_blocks[3], stride=2), *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2))

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int=1):
        """starts with conv layer then has `num_blocks` `ResLayer`"""
        return [BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act='lrelu'), *[ResLayer(in_channels * 2) for _ in range(num_blocks)]]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(*[BaseConv(in_filters, filters_list[0], 1, stride=1, act='lrelu'), BaseConv(filters_list[0], filters_list[1], 3, stride=1, act='lrelu'), SPPBottleneck(in_channels=filters_list[1], out_channels=filters_list[0], activation='lrelu'), BaseConv(filters_list[0], filters_list[1], 3, stride=1, act='lrelu'), BaseConv(filters_list[1], filters_list[0], 1, stride=1, act='lrelu')])
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs['stem'] = x
        x = self.dark2(x)
        outputs['dark2'] = x
        x = self.dark3(x)
        outputs['dark3'] = x
        x = self.dark4(x)
        outputs['dark4'] = x
        x = self.dark5(x)
        outputs['dark5'] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act='silu'):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act='silu'):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act='silu'):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act='silu'):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return self.conv(x)


class CSPDarknet(nn.Module):

    def __init__(self, dep_mul, wid_mul, out_features=('dark3', 'dark4', 'dark5'), depthwise=False, act='silu'):
        super().__init__()
        assert out_features, 'please provide output features of Darknet'
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)
        self.stem = Focus(3, base_channels, ksize=3, act=act)
        self.dark2 = nn.Sequential(Conv(base_channels, base_channels * 2, 3, 2, act=act), CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act))
        self.dark3 = nn.Sequential(Conv(base_channels * 2, base_channels * 4, 3, 2, act=act), CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act))
        self.dark4 = nn.Sequential(Conv(base_channels * 4, base_channels * 8, 3, 2, act=act), CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act))
        self.dark5 = nn.Sequential(Conv(base_channels * 8, base_channels * 16, 3, 2, act=act), SPPBottleneck(base_channels * 16, base_channels * 16, activation=act), CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act))

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs['stem'] = x
        x = self.dark2(x)
        outputs['dark2'] = x
        x = self.dark3(x)
        outputs['dark3'] = x
        x = self.dark4(x)
        outputs['dark4'] = x
        x = self.dark5(x)
        outputs['dark5'] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknetP6(nn.Module):

    def __init__(self, dep_mul, wid_mul, out_features=('dark3', 'dark4', 'dark5', 'dark6'), depthwise=False, act='silu'):
        super().__init__()
        assert out_features, 'please provide output features of Darknet'
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)
        self.stem = Focus(3, base_channels, ksize=3, act=act)
        self.dark2 = nn.Sequential(Conv(base_channels, base_channels * 2, 3, 2, act=act), CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act))
        self.dark3 = nn.Sequential(Conv(base_channels * 2, base_channels * 4, 3, 2, act=act), CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act))
        self.dark4 = nn.Sequential(Conv(base_channels * 4, base_channels * 8, 3, 2, act=act), CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act))
        self.dark5 = nn.Sequential(Conv(base_channels * 8, base_channels * 16, 3, 2, act=act), CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, depthwise=depthwise, act=act))
        self.dark6 = nn.Sequential(Conv(base_channels * 16, base_channels * 16, 3, 2, act=act), SPPBottleneck(base_channels * 16, base_channels * 16, activation=act), CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act))

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs['stem'] = x
        x = self.dark2(x)
        outputs['dark2'] = x
        x = self.dark3(x)
        outputs['dark3'] = x
        x = self.dark4(x)
        outputs['dark4'] = x
        x = self.dark5(x)
        outputs['dark5'] = x
        x = self.dark6(x)
        outputs['dark6'] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class Darknet53(nn.Module):

    def __init__(self, in_channels=3, stem_out_channel=32, out_features=('stage3', 'stage4', 'stage5')):
        super().__init__()
        self.out_features = out_features
        self.blocknum = [1, 2, 8, 8, 4]
        self.stem = nn.Sequential(ConvBnLeaky(in_channels, out_channels=stem_out_channel, ksize=3, stride=1))
        self.stage1 = nn.Sequential(*self.make_group_layer(in_channels=stem_out_channel, block_num=self.blocknum[0], stride=2))
        self.stage2 = nn.Sequential(*self.make_group_layer(in_channels=2 * stem_out_channel, block_num=self.blocknum[1], stride=2))
        self.stage3 = nn.Sequential(*self.make_group_layer(in_channels=4 * stem_out_channel, block_num=self.blocknum[2], stride=2))
        self.stage4 = nn.Sequential(*self.make_group_layer(in_channels=8 * stem_out_channel, block_num=self.blocknum[3], stride=2))
        self.stage5 = nn.Sequential(*self.make_group_layer(in_channels=16 * stem_out_channel, block_num=self.blocknum[4], stride=2))

    def make_group_layer(self, in_channels, block_num, stride=1):
        return [ConvBnLeaky(in_channels=in_channels, out_channels=in_channels * 2, ksize=3, stride=stride), *[Res_unit(in_channels * 2) for _ in range(block_num)]]

    def forward(self, x):
        output = {}
        x = self.stem(x)
        output['stem'] = x
        x = self.stage1(x)
        output['stage1'] = x
        x = self.stage2(x)
        output['stage2'] = x
        x = self.stage3(x)
        output['stage3'] = x
        x = self.stage4(x)
        output['stage4'] = x
        x = self.stage5(x)
        output['stage5'] = x
        return {k: v for k, v in output.items() if k in self.out_features}


class IOUloss(nn.Module):

    def __init__(self, reduction='none', loss_type='iou'):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(pred[:, :2] - pred[:, 2:] / 2, target[:, :2] - target[:, 2:] / 2)
        br = torch.min(pred[:, :2] + pred[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2)
        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = area_i / (area_u + 1e-16)
        if self.loss_type == 'iou':
            loss = 1 - iou ** 2
        elif self.loss_type == 'giou':
            c_tl = torch.min(pred[:, :2] - pred[:, 2:] / 2, target[:, :2] - target[:, 2:] / 2)
            c_br = torch.max(pred[:, :2] + pred[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2)
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(self, depth=1.0, width=1.0, in_features=('dark3', 'dark4', 'dark5'), in_channels=[256, 512, 1024], depthwise=False, act='silu'):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(int(2 * in_channels[1] * width), int(in_channels[1] * width), round(3 * depth), False, depthwise=depthwise, act=act)
        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(int(2 * in_channels[0] * width), int(in_channels[0] * width), round(3 * depth), False, depthwise=depthwise, act=act)
        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.C3_n3 = CSPLayer(int(2 * in_channels[0] * width), int(in_channels[1] * width), round(3 * depth), False, depthwise=depthwise, act=act)
        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        self.C3_n4 = CSPLayer(int(2 * in_channels[1] * width), int(in_channels[2] * width), round(3 * depth), False, depthwise=depthwise, act=act)

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features
        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat([f_out0, x1], 1)
        f_out0 = self.C3_p4(f_out0)
        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = torch.cat([f_out1, x2], 1)
        pan_out2 = self.C3_p3(f_out1)
        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = torch.cat([p_out1, fpn_out1], 1)
        pan_out1 = self.C3_n3(p_out1)
        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], 1)
        pan_out0 = self.C3_n4(p_out0)
        outputs = pan_out2, pan_out1, pan_out0
        return outputs


class Attention_msa(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, scale=25):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = scale
        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, return_attention=False, ave=True, sim_thresh=0.75, use_mask=False):
        B, N, C = x_cls.shape
        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]
        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls, dim=-1, keepdim=True)
        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)
        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)
        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)
        if use_mask:
            cls_score_mask = (cls_score > cls_score.transpose(-2, -1) - 0.1).type_as(cls_score)
            fg_score_mask = (fg_score > fg_score.transpose(-2, -1) - 0.1).type_as(fg_score)
        else:
            cls_score_mask = fg_score_mask = 1
        attn_cls = q_cls @ k_cls.transpose(-2, -1) * self.scale * cls_score * cls_score_mask
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)
        attn_reg = q_reg @ k_reg.transpose(-2, -1) * self.scale * fg_score * fg_score_mask
        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)
        attn = (attn_reg + attn_cls) / 2
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)
        x_ori = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)
        x_cls = torch.cat([x, x_ori], dim=-1)
        if ave:
            ones_matrix = torch.ones(attn.shape[2:])
            zero_matrix = torch.zeros(attn.shape[2:])
            attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads
            sim_mask = torch.where(attn_cls_raw > sim_thresh, ones_matrix, zero_matrix)
            sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads
            sim_round2 = torch.softmax(sim_attn, dim=-1)
            sim_round2 = sim_mask * sim_round2 / torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True)
            return x_cls, None, sim_round2
        else:
            return x_cls, None, None


class MSA_yolov(nn.Module):

    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0.0, scale=25):
        super().__init__()
        self.msa = Attention_msa(dim, num_heads, qkv_bias, attn_drop, scale=scale)
        self.linear1 = nn.Linear(2 * dim, 2 * dim)
        self.linear2 = nn.Linear(4 * dim, out_dim)

    def find_similar_round2(self, features, sort_results):
        key_feature = features[0]
        support_feature = features[0]
        if not self.training:
            sort_results = sort_results
        soft_sim_feature = sort_results @ support_feature
        cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1)
        return cls_feature

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, sim_thresh=0.75, ave=True, use_mask=False):
        trans_cls, trans_reg, sim_round2 = self.msa(x_cls, x_reg, cls_score, fg_score, sim_thresh=sim_thresh, ave=ave, use_mask=use_mask)
        msa = self.linear1(trans_cls)
        msa = self.find_similar_round2(msa, sim_round2)
        out = self.linear2(msa)
        return out


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2, bboxes_b[:, :2] - bboxes_b[:, 2:] / 2)
        br = torch.min(bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2, bboxes_b[:, :2] + bboxes_b[:, 2:] / 2)
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en
    return area_i / (area_a[:, None] + area_b - area_i)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area, iou


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        if not image_pred.size(0):
            continue
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue
        if class_agnostic:
            nms_out_index = torchvision.ops.nms(detections[:, :4], detections[:, 4] * detections[:, 5], nms_thre)
        else:
            nms_out_index = torchvision.ops.batched_nms(detections[:, :4], detections[:, 4] * detections[:, 5], detections[:, 6], nms_thre)
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
    return output


class YOLOXHead(nn.Module):

    def __init__(self, num_classes, width=1.0, strides=[8, 16, 32], in_channels=[256, 512, 1024], act='silu', depthwise=False, heads=4, drop=0.0, use_score=True, defualt_p=30, sim_thresh=0.75, pre_nms=0.75, ave=True, defulat_pre=750, test_conf=0.001, use_mask=False):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()
        self.Afternum = defualt_p
        self.Prenum = defulat_pre
        self.simN = defualt_p
        self.nms_thresh = pre_nms
        self.n_anchors = 1
        self.use_score = use_score
        self.num_classes = num_classes
        self.decode_in_inference = True
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.cls_convs2 = nn.ModuleList()
        self.width = int(256 * width)
        self.trans = MSA_yolov(dim=self.width, out_dim=4 * self.width, num_heads=heads, attn_drop=drop)
        self.stems = nn.ModuleList()
        self.linear_pred = nn.Linear(int(4 * self.width), num_classes + 1)
        self.sim_thresh = sim_thresh
        self.ave = ave
        self.use_mask = use_mask
        Conv = DWConv if depthwise else BaseConv
        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width), ksize=1, stride=1, act=act))
            self.cls_convs.append(nn.Sequential(*[Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act), Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)]))
            self.cls_convs2.append(nn.Sequential(*[Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act), Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)]))
            self.reg_convs.append(nn.Sequential(*[Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act), Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)]))
            self.cls_preds.append(nn.Conv2d(in_channels=int(256 * width), out_channels=self.n_anchors * self.num_classes, kernel_size=1, stride=1, padding=0))
            self.reg_preds.append(nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1, padding=0))
            self.obj_preds.append(nn.Conv2d(in_channels=int(256 * width), out_channels=self.n_anchors * 1, kernel_size=1, stride=1, padding=0))
        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction='none')
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.iou_loss = IOUloss(reduction='none')
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None, nms_thresh=0.5):
        outputs = []
        outputs_decode = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        before_nms_features = []
        before_nms_regf = []
        for k, (cls_conv, cls_conv2, reg_conv, stride_this_level, x) in enumerate(zip(self.cls_convs, self.cls_convs2, self.reg_convs, self.strides, xin)):
            x = self.stems[k](x)
            reg_feat = reg_conv(x)
            cls_feat = cls_conv(x)
            cls_feat2 = cls_conv2(x)
            obj_output = self.obj_preds[k](reg_feat)
            reg_output = self.reg_preds[k](reg_feat)
            cls_output = self.cls_preds[k](cls_feat)
            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output_decode = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
                output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(xin[0]))
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(batch_size, self.n_anchors, 4, hsize, wsize)
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                    origin_preds.append(reg_output.clone())
                outputs.append(output)
                before_nms_features.append(cls_feat2)
                before_nms_regf.append(reg_feat)
            else:
                output_decode = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
                before_nms_features.append(cls_feat2)
                before_nms_regf.append(reg_feat)
            outputs_decode.append(output_decode)
        self.hw = [x.shape[-2:] for x in outputs_decode]
        outputs_decode = torch.cat([x.flatten(start_dim=2) for x in outputs_decode], dim=2).permute(0, 2, 1)
        decode_res = self.decode_outputs(outputs_decode, dtype=xin[0].type())
        pred_result, pred_idx = self.postpro_woclass(decode_res, num_classes=self.num_classes, nms_thre=self.nms_thresh, topK=self.Afternum)
        if not self.training and imgs.shape[0] == 1:
            return self.postprocess_single_img(pred_result, self.num_classes)
        cls_feat_flatten = torch.cat([x.flatten(start_dim=2) for x in before_nms_features], dim=2).permute(0, 2, 1)
        reg_feat_flatten = torch.cat([x.flatten(start_dim=2) for x in before_nms_regf], dim=2).permute(0, 2, 1)
        features_cls, features_reg, cls_scores, fg_scores = self.find_feature_score(cls_feat_flatten, pred_idx, reg_feat_flatten, imgs, pred_result)
        features_reg = features_reg.unsqueeze(0)
        features_cls = features_cls.unsqueeze(0)
        if not self.training:
            cls_scores = cls_scores
            fg_scores = fg_scores
        if self.use_score:
            trans_cls = self.trans(features_cls, features_reg, cls_scores, fg_scores, sim_thresh=self.sim_thresh, ave=self.ave, use_mask=self.use_mask)
        else:
            trans_cls = self.trans(features_cls, features_reg, None, None, sim_thresh=self.sim_thresh, ave=self.ave)
        fc_output = self.linear_pred(trans_cls)
        fc_output = torch.reshape(fc_output, [outputs_decode.shape[0], -1, self.num_classes + 1])[:, :, :-1]
        if self.training:
            return self.get_losses(imgs, x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1), origin_preds, dtype=xin[0].dtype, refined_cls=fc_output, idx=pred_idx, pred_res=pred_result)
        else:
            class_conf, class_pred = torch.max(fc_output, -1, keepdim=False)
            result, result_ori = postprocess(copy.deepcopy(pred_result), self.num_classes, fc_output, nms_thre=nms_thresh)
            return result, result_ori

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]
        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid
        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, self.n_anchors * hsize * wsize, -1)
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype, flevel=0):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))
        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def find_feature_score(self, features, idxs, reg_features, imgs=None, predictions=None, roi_features=None):
        features_cls = []
        features_reg = []
        cls_scores = []
        fg_scores = []
        for i, feature in enumerate(features):
            features_cls.append(feature[idxs[i][:self.simN]])
            features_reg.append(reg_features[i, idxs[i][:self.simN]])
            cls_scores.append(predictions[i][:self.simN, 5])
            fg_scores.append(predictions[i][:self.simN, 4])
        features_cls = torch.cat(features_cls)
        features_reg = torch.cat(features_reg)
        cls_scores = torch.cat(cls_scores)
        fg_scores = torch.cat(fg_scores)
        return features_cls, features_reg, cls_scores, fg_scores

    def get_losses(self, imgs, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype, refined_cls, idx, pred_res):
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)
        cls_preds = outputs[:, :, 5:]
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)
        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        ref_targets = []
        num_fg = 0.0
        num_gts = 0.0
        ref_masks = []
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                ref_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))
                ref_target[:, -1] = 1
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                try:
                    gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts, cls_preds, bbox_preds, obj_preds, labels, imgs)
                except RuntimeError:
                    logger.error('OOM RuntimeError is raised due to the huge memory cost during label assignment.                            CPU mode is applied in this batch. If you want to avoid this issue,                            try to reduce the batch size or image size.')
                    torch.cuda.empty_cache()
                    gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts, cls_preds, bbox_preds, obj_preds, labels, imgs, 'cpu')
                torch.cuda.empty_cache()
                num_fg += num_fg_img
                cls_target_onehot = F.one_hot(gt_matched_classes, self.num_classes)
                cls_target = cls_target_onehot * pred_ious_this_matching.unsqueeze(-1)
                fg_idx = torch.where(fg_mask)[0]
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(outputs.new_zeros((num_fg_img, 4)), gt_bboxes_per_image[matched_gt_inds], expanded_strides[0][fg_mask], x_shifts=x_shifts[0][fg_mask], y_shifts=y_shifts[0][fg_mask])
                ref_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))
                fg = 0
                gt_xyxy = box_cxcywh_to_xyxy(torch.tensor(reg_target))
                pred_box = pred_res[batch_idx][:, :4]
                cost_giou, iou = generalized_box_iou(pred_box, gt_xyxy)
                max_iou = torch.max(iou, dim=-1)
                for ele_idx, ele in enumerate(idx[batch_idx]):
                    loc = torch.where(fg_idx == ele)[0]
                    if len(loc):
                        ref_target[ele_idx, :self.num_classes] = cls_target[loc, :]
                        fg += 1
                        continue
                    if max_iou.values[ele_idx] >= 0.6:
                        max_idx = int(max_iou.indices[ele_idx])
                        ref_target[ele_idx, :self.num_classes] = cls_target_onehot[max_idx, :] * max_iou.values[ele_idx]
                        fg += 1
                    else:
                        ref_target[ele_idx, -1] = 1 - max_iou.values[ele_idx]
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target)
            fg_masks.append(fg_mask)
            ref_targets.append(ref_target[:, :self.num_classes])
            ref_masks.append(ref_target[:, -1] == 0)
            if self.use_l1:
                l1_targets.append(l1_target)
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        ref_targets = torch.cat(ref_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        ref_masks = torch.cat(ref_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
        num_fg = max(num_fg, 1)
        loss_iou = self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets).sum() / num_fg
        loss_obj = self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets).sum() / num_fg
        loss_cls = self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets).sum() / num_fg
        loss_ref = self.bcewithlog_loss(refined_cls.view(-1, self.num_classes)[ref_masks], ref_targets[ref_masks]).sum() / num_fg
        if self.use_l1:
            loss_l1 = self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets).sum() / num_fg
        else:
            loss_l1 = 0.0
        reg_weight = 3.0
        loss = reg_weight * loss_iou + loss_obj + 2 * loss_ref + loss_l1 + loss_cls
        return loss, reg_weight * loss_iou, loss_obj, 2 * loss_ref, loss_l1, num_fg / max(num_gts, 1)

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-08):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(self, batch_idx, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts, cls_preds, bbox_preds, obj_preds, labels, imgs, mode='gpu'):
        if mode == 'cpu':
            None
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt)
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]
        if mode == 'cpu':
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        gt_cls_per_image = F.one_hot(gt_classes, self.num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-08)
        if mode == 'cpu':
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()
        with torch.amp.autocast(enabled=False):
            cls_preds_ = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction='none').sum(-1)
        del cls_preds_
        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * ~is_in_boxes_and_center
        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        if mode == 'cpu':
            gt_matched_classes = gt_matched_classes
            fg_mask = fg_mask
            pred_ious_this_matching = pred_ious_this_matching
            matched_gt_inds = matched_gt_inds
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (x_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        y_centers_per_image = (y_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        center_radius = 4.5
        gt_bboxes_per_image_l = gt_bboxes_per_image[:, 0].unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = gt_bboxes_per_image[:, 0].unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = gt_bboxes_per_image[:, 1].unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = gt_bboxes_per_image[:, 1].unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost)
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[gt_idx][pos_idx] = 1.0
        del topk_ious, dynamic_ks, pos_idx
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()
        fg_mask[fg_mask.clone()] = fg_mask_inboxes
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def postpro_woclass(self, prediction, num_classes, nms_thre=0.75, topK=75, features=None):
        """

        Args:
            prediction: [batch,feature_num,5+clsnum]
            num_classes:
            conf_thre:
            conf_thre_high:
            nms_thre:

        Returns:
            [batch,topK,5+clsnum]
        """
        self.topK = topK
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
        output = [None for _ in range(len(prediction))]
        output_index = [None for _ in range(len(prediction))]
        features_list = []
        for i, image_pred in enumerate(prediction):
            if not image_pred.size(0):
                continue
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5:5 + num_classes]), 1)
            conf_score = image_pred[:, 4]
            top_pre = torch.topk(conf_score, k=self.Prenum)
            sort_idx = top_pre.indices[:self.Prenum]
            detections_temp = detections[sort_idx, :]
            nms_out_index = torchvision.ops.batched_nms(detections_temp[:, :4], detections_temp[:, 4] * detections_temp[:, 5], detections_temp[:, 6], nms_thre)
            topk_idx = sort_idx[nms_out_index[:self.topK]]
            output[i] = detections[topk_idx, :]
            output_index[i] = topk_idx
        return output, output_index

    def postprocess_single_img(self, prediction, num_classes, conf_thre=0.001, nms_thre=0.5):
        output_ori = [None for _ in range(len(prediction))]
        prediction_ori = copy.deepcopy(prediction)
        for i, detections in enumerate(prediction):
            if not detections.size(0):
                continue
            detections_ori = prediction_ori[i]
            conf_mask = (detections_ori[:, 4] * detections_ori[:, 5] >= conf_thre).squeeze()
            detections_ori = detections_ori[conf_mask]
            nms_out_index = torchvision.ops.batched_nms(detections_ori[:, :4], detections_ori[:, 4] * detections_ori[:, 5], detections_ori[:, 6], nms_thre)
            detections_ori = detections_ori[nms_out_index]
            output_ori[i] = detections_ori
        return output_ori, output_ori


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)
        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        fpn_outs = self.backbone(x)
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, x)
            outputs = {'total_loss': loss, 'iou_loss': iou_loss, 'l1_loss': l1_loss, 'conf_loss': conf_loss, 'cls_loss': cls_loss, 'num_fg': num_fg}
        else:
            outputs = self.head(fpn_outs)
        return outputs


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Attention_msa_visual(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, scale=25):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = 30
        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, img=None, pred=None):
        B, N, C = x_cls.shape
        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]
        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls, dim=-1, keepdim=True)
        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)
        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)
        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)
        attn_cls = q_cls @ k_cls.transpose(-2, -1) * self.scale * cls_score
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)
        attn_reg = q_reg @ k_reg.transpose(-2, -1) * self.scale * fg_score
        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)
        attn = (attn_cls_raw * 25).softmax(dim=-1)
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)
        x_ori = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)
        x_cls = torch.cat([x, x_ori], dim=-1)
        ones_matrix = torch.ones(attn.shape[2:])
        zero_matrix = torch.zeros(attn.shape[2:])
        attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads
        sim_mask = torch.where(attn_cls_raw > 0.75, ones_matrix, zero_matrix)
        sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads
        sim_round2 = torch.softmax(sim_attn, dim=-1)
        sim_round2 = sim_mask * sim_round2 / torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True)
        attn_total = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads
        visual_sim(attn_total, img, 30, pred, attn_cls_raw)
        return x_cls, None, sim_round2


class Attention_msa_online(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, scale=25):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = scale
        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, return_attention=False, ave=True):
        B, N, C = x_cls.shape
        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]
        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls, dim=-1, keepdim=True)
        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)
        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)
        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)
        attn_cls = q_cls @ k_cls.transpose(-2, -1) * self.scale * cls_score
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)
        attn_reg = q_reg @ k_reg.transpose(-2, -1) * self.scale * fg_score
        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)
        attn = (attn_reg + attn_cls) / 2
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)
        x_ori = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)
        x_cls = torch.cat([x, x_ori], dim=-1)
        if ave:
            ones_matrix = torch.ones(attn.shape[2:])
            zero_matrix = torch.zeros(attn.shape[2:])
            attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads
            sim_mask = torch.where(attn_cls_raw > 0.75, ones_matrix, zero_matrix)
            sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads
            sim_round2 = torch.softmax(sim_attn, dim=-1)
            sim_round2 = sim_mask * sim_round2 / torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True)
            return x_cls, None, sim_round2
        else:
            return x_cls


class MSA_yolov_visual(nn.Module):

    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0.0, scale=25):
        super().__init__()
        self.msa = Attention_msa_visual(dim, num_heads, qkv_bias, attn_drop, scale=scale)
        self.linear1 = nn.Linear(2 * dim, 2 * dim)
        self.linear2 = nn.Linear(4 * dim, out_dim)

    def ave_pooling_over_ref(self, features, sort_results):
        key_feature = features[0]
        support_feature = features[0]
        if not self.training:
            sort_results = sort_results
        soft_sim_feature = sort_results @ support_feature
        cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1)
        return cls_feature

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, img=None, pred=None):
        trans_cls, trans_reg, sim_round2 = self.msa(x_cls, x_reg, cls_score, fg_score, img, pred)
        msa = self.linear1(trans_cls)
        ave = self.ave_pooling_over_ref(msa, sim_round2)
        out = self.linear2(ave)
        return out


class MSA_yolov_online(nn.Module):

    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0.0, scale=25):
        super().__init__()
        self.msa = Attention_msa_online(dim, num_heads, qkv_bias, attn_drop, scale=scale)
        self.linear1 = nn.Linear(2 * dim, 2 * dim)
        self.linear2 = nn.Linear(4 * dim, out_dim)

    def ave_pooling_over_ref(self, features, sort_results):
        key_feature = features[0]
        support_feature = features[0]
        if not self.training:
            sort_results = sort_results
        soft_sim_feature = sort_results @ support_feature
        cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1)
        return cls_feature

    def compute_geo_sim(self, key_preds, ref_preds):
        key_boxes = key_preds[:, :4]
        ref_boxes = ref_preds[:, :4]
        cost_giou, iou = generalized_box_iou(key_boxes, ref_boxes)
        return iou

    def local_agg(self, features, local_results, boxes, cls_score, fg_score):
        local_features = local_results['msa']
        local_features_n = local_features / torch.norm(local_features, dim=-1, keepdim=True)
        features_n = features / torch.norm(features, dim=-1, keepdim=True)
        cos_sim = features_n @ local_features_n.transpose(0, 1)
        geo_sim = self.compute_geo_sim(boxes, local_results['boxes'])
        N = local_results['cls_scores'].shape[0]
        M = cls_score.shape[0]
        pre_scores = cls_score * fg_score
        pre_scores = torch.reshape(pre_scores, [-1, 1]).repeat(1, N)
        other_scores = local_results['cls_scores'] * local_results['reg_scores']
        other_scores = torch.reshape(other_scores, [1, -1]).repeat(M, 1)
        ones_matrix = torch.ones([M, N])
        zero_matrix = torch.zeros([M, N])
        thresh_map = torch.where(other_scores - pre_scores > -0.3, ones_matrix, zero_matrix)
        local_sim = torch.softmax(25 * cos_sim * thresh_map, dim=-1) * geo_sim
        local_sim = local_sim / torch.sum(local_sim, dim=-1, keepdim=True)
        local_sim = local_sim
        sim_features = local_sim @ local_features
        return (sim_features + features) / 2

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, other_result={}, boxes=None, simN=30):
        trans_cls, trans_reg, sim_round2 = self.msa(x_cls, x_reg, cls_score, fg_score)
        msa = self.linear1(trans_cls)
        ave = self.ave_pooling_over_ref(msa, sim_round2)
        out = self.linear2(ave)
        if other_result != [] and other_result['local_results'] != []:
            lout = self.local_agg(out[:simN], other_result['local_results'], boxes[:simN], cls_score[:simN], fg_score[:simN])
            return lout, out
        return out, out


class YOLOFPN(nn.Module):
    """
    YOLOFPN module. Darknet 53 is the default backbone of this model.
    """

    def __init__(self, depth=53, in_features=['dark3', 'dark4', 'dark5']):
        super().__init__()
        self.backbone = Darknet(depth)
        self.in_features = in_features
        self.out1_cbl = self._make_cbl(512, 256, 1)
        self.out1 = self._make_embedding([256, 512], 512 + 256)
        self.out2_cbl = self._make_cbl(256, 128, 1)
        self.out2 = self._make_embedding([128, 256], 256 + 128)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _make_cbl(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=1, act='lrelu')

    def _make_embedding(self, filters_list, in_filters):
        m = nn.Sequential(*[self._make_cbl(in_filters, filters_list[0], 1), self._make_cbl(filters_list[0], filters_list[1], 3), self._make_cbl(filters_list[1], filters_list[0], 1), self._make_cbl(filters_list[0], filters_list[1], 3), self._make_cbl(filters_list[1], filters_list[0], 1)])
        return m

    def load_pretrained_model(self, filename='./weights/darknet53.mix.pth'):
        with open(filename, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
        None
        self.backbone.load_state_dict(state_dict)

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): input image.

        Returns:
            Tuple[Tensor]: FPN output features..
        """
        out_features = self.backbone(inputs)
        x2, x1, x0 = [out_features[f] for f in self.in_features]
        x1_in = self.out1_cbl(x0)
        x1_in = self.upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out_dark4 = self.out1(x1_in)
        x2_in = self.out2_cbl(out_dark4)
        x2_in = self.upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out_dark3 = self.out2(x2_in)
        outputs = out_dark3, out_dark4, x0
        return outputs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'ksize': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CSPDarknet,
     lambda: ([], {'dep_mul': 4, 'wid_mul': 4}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     False),
    (CSPDarknetP6,
     lambda: ([], {'dep_mul': 4, 'wid_mul': 4}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     False),
    (CSPLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DWConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'ksize': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Focus,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IOUloss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MSA_yolov,
     lambda: ([], {'dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (ResLayer,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SPPBottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (YOLOFPN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_YuHengsss_YOLOV(_paritybench_base):
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

