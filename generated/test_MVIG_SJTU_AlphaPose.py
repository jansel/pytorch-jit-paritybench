import sys
_module = sys.modules[__name__]
del sys
matching = _module
parallel_process = _module
poseflow_infer = _module
utils = _module
alphapose = _module
datasets = _module
coco_det = _module
concat_dataset = _module
custom = _module
mpii = _module
mscoco = _module
models = _module
builder = _module
fastpose = _module
fastpose_duc = _module
fastpose_duc_dense = _module
hrnet = _module
DUC = _module
PixelUnshuffle = _module
Resnet = _module
SE_Resnet = _module
SE_module = _module
ShuffleResnet = _module
DCN = _module
dcn = _module
deform_conv = _module
deform_pool = _module
simplepose = _module
opt = _module
bbox = _module
config = _module
detector = _module
env = _module
file_detector = _module
logger = _module
metrics = _module
pPose_nms = _module
presets = _module
simple_transform = _module
registry = _module
roi_align = _module
roi_align = _module
transforms = _module
vis = _module
webcam_detector = _module
writer = _module
version = _module
apis = _module
effdet_api = _module
effdet_cfg = _module
effdet = _module
anchors = _module
bench = _module
efficientdet = _module
helpers = _module
object_detection = _module
argmax_matcher = _module
box_coder = _module
box_list = _module
faster_rcnn_box_coder = _module
matcher = _module
region_similarity_calculator = _module
target_assigner = _module
nms = _module
nms_wrapper = _module
tracker = _module
models = _module
preprocess = _module
basetrack = _module
multitracker = _module
evaluation = _module
io = _module
kalman_filter = _module
log = _module
parse_config = _module
timer = _module
utils = _module
visualization = _module
tracker_api = _module
tracker_cfg = _module
yolo = _module
cam_demo = _module
darknet = _module
detect = _module
preprocess = _module
util = _module
video_demo = _module
video_demo_half = _module
yolo_api = _module
yolo_cfg = _module
demo_inference = _module
train = _module
validate = _module
setup = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch.nn as nn


import torch


import torch.nn.functional as F


from torch import nn


import math


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.nn.modules.utils import _pair


import numpy as np


import inspect


from abc import ABC


from abc import abstractmethod


import collections


from torchvision.ops.boxes import batched_nms


import logging


from collections import OrderedDict


from typing import List


from typing import Optional


from torch.nn.functional import one_hot


from collections import defaultdict


import time


from torch.autograd import Variable


from collections import deque


import itertools


import random


import torch.utils.data


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


SPPE = Registry('sppe')


@SPPE.register_module
class FastPose(nn.Module):
    conv_dim = 128

    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(FastPose, self).__init__()
        self._preset_cfg = cfg['PRESET']
        if 'DCN' in cfg.keys():
            stage_with_dcn = cfg['STAGE_WITH_DCN']
            dcn = cfg['DCN']
            self.preact = SEResnet(f"resnet{cfg['NUM_LAYERS']}", dcn=dcn, stage_with_dcn=stage_with_dcn)
        else:
            self.preact = SEResnet(f"resnet{cfg['NUM_LAYERS']}")
        import torchvision.models as tm
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items() if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)
        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2, norm_layer=norm_layer)
        self.duc2 = DUC(256, 512, upscale_factor=2, norm_layer=norm_layer)
        self.conv_out = nn.Conv2d(self.conv_dim, self._preset_cfg['NUM_JOINTS'], kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)
        out = self.conv_out(out)
        return out

    def _initialize(self):
        for m in self.conv_out.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)


@SPPE.register_module
class FastPose_DUC(nn.Module):
    conv_dim = 256

    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(FastPose_DUC, self).__init__()
        self._preset_cfg = cfg['PRESET']
        if cfg['BACKBONE'] == 'shuffle':
            None
            backbone = ShuffleResnet
        elif cfg['BACKBONE'] == 'se-resnet':
            None
            backbone = SEResnet
        else:
            None
            backbone = ResNet
        if 'DCN' in cfg.keys():
            stage_with_dcn = cfg['STAGE_WITH_DCN']
            dcn = cfg['DCN']
            self.preact = backbone(f"resnet{cfg['NUM_LAYERS']}", dcn=dcn, stage_with_dcn=stage_with_dcn)
        else:
            self.preact = backbone(f"resnet{cfg['NUM_LAYERS']}")
        import torchvision.models as tm
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items() if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)
        self.norm_layer = norm_layer
        stage1_cfg = cfg['STAGE1']
        stage2_cfg = cfg['STAGE2']
        stage3_cfg = cfg['STAGE3']
        self.duc1 = self._make_duc_stage(stage1_cfg, 2048, 1024)
        self.duc2 = self._make_duc_stage(stage2_cfg, 1024, 512)
        self.duc3 = self._make_duc_stage(stage3_cfg, 512, self.conv_dim)
        self.conv_out = nn.Conv2d(self.conv_dim, self._preset_cfg['NUM_JOINTS'], kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.preact(x)
        out = self.duc1(out)
        out = self.duc2(out)
        out = self.duc3(out)
        out = self.conv_out(out)
        return out

    def _make_duc_stage(self, layer_config, inplanes, outplanes):
        layers = []
        shuffle = nn.PixelShuffle(2)
        inplanes //= 4
        layers.append(shuffle)
        for i in range(layer_config.NUM_CONV - 1):
            conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, bias=False)
            norm_layer = self.norm_layer(inplanes, momentum=0.1)
            relu = nn.ReLU(inplace=True)
            layers += [conv, norm_layer, relu]
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1, bias=False)
        norm_layer = self.norm_layer(outplanes, momentum=0.1)
        relu = nn.ReLU(inplace=True)
        layers += [conv, norm_layer, relu]
        return nn.Sequential(*layers)

    def _initialize(self):
        for m in self.conv_out.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)


@SPPE.register_module
class FastPose_DUC_Dense(nn.Module):
    conv_dim = 256

    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(FastPose_DUC_Dense, self).__init__()
        self._preset_cfg = cfg['PRESET']
        if cfg['BACKBONE'] == 'shuffle':
            None
            backbone = ShuffleResnet
        elif cfg['BACKBONE'] == 'se-resnet':
            None
            backbone = SEResnet
        else:
            None
            backbone = ResNet
        if 'DCN' in cfg.keys():
            stage_with_dcn = cfg['STAGE_WITH_DCN']
            dcn = cfg['DCN']
            self.preact = backbone(f"resnet{cfg['NUM_LAYERS']}", dcn=dcn, stage_with_dcn=stage_with_dcn)
        else:
            self.preact = backbone(f"resnet{cfg['NUM_LAYERS']}")
        for m in self.preact.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, 0, 1)
                nn.init.constant_(m.bias, 0)
        import torchvision.models as tm
        if cfg['NUM_LAYERS'] == 152:
            """ Load pretrained model """
            x = tm.resnet152(pretrained=True)
        elif cfg['NUM_LAYERS'] == 101:
            """ Load pretrained model """
            x = tm.resnet101(pretrained=True)
        elif cfg['NUM_LAYERS'] == 50:
            x = tm.resnet50(pretrained=True)
        elif cfg['NUM_LAYERS'] == 18:
            x = tm.resnet18(pretrained=True)
        else:
            raise NotImplementedError
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items() if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)
        self.norm_layer = norm_layer
        stage1_cfg = cfg['STAGE1']
        stage2_cfg = cfg['STAGE2']
        stage3_cfg = cfg['STAGE3']
        duc1 = self._make_duc_stage(stage1_cfg, 2048, 1024)
        duc2 = self._make_duc_stage(stage2_cfg, 1024, 512)
        duc3 = self._make_duc_stage(stage3_cfg, 512, self.conv_dim)
        self.duc = nn.Sequential(duc1, duc2, duc3)
        duc1_dense = self._make_duc_stage(stage1_cfg, 2048, 1024)
        duc2_dense = self._make_duc_stage(stage2_cfg, 1024, 512)
        duc3_dense = self._make_duc_stage(stage3_cfg, 512, self.conv_dim)
        self.duc_dense = nn.Sequential(duc1_dense, duc2_dense, duc3_dense)
        self.conv_out = nn.Conv2d(self.conv_dim, self._preset_cfg['NUM_JOINTS'], kernel_size=3, stride=1, padding=1)
        self.conv_out_dense = nn.Conv2d(self.conv_dim, self._preset_cfg['NUM_JOINTS_DENSE'] - self._preset_cfg['NUM_JOINTS'], kernel_size=3, stride=1, padding=1)
        for params in self.preact.parameters():
            params.requires_grad = False
        for params in self.duc.parameters():
            params.requires_grad = False

    def forward(self, x):
        bk_out = self.preact(x)
        out = self.duc(bk_out)
        out_dense = self.duc_dense(bk_out)
        out = self.conv_out(out)
        out_dense = self.conv_out_dense(out_dense)
        out = torch.cat((out, out_dense), 1)
        return out

    def _make_duc_stage(self, layer_config, inplanes, outplanes):
        layers = []
        shuffle = nn.PixelShuffle(2)
        inplanes //= 4
        layers.append(shuffle)
        for i in range(layer_config.NUM_CONV - 1):
            conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, bias=False)
            norm_layer = self.norm_layer(inplanes, momentum=0.1)
            relu = nn.ReLU(inplace=True)
            layers += [conv, norm_layer, relu]
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1, bias=False)
        norm_layer = self.norm_layer(outplanes, momentum=0.1)
        relu = nn.ReLU(inplace=True)
        layers += [conv, norm_layer, relu]
        return nn.Sequential(*layers)

    def _initialize(self):
        for m in self.duc.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, 0, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.conv_out.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        for m in self.duc_dense.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, 0, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.conv_out_dense.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)


BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
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
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
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


class HighResolutionModule(nn.Module):

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
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
            downsample = nn.Sequential(nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
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
                    fuse_layer.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False), nn.BatchNorm2d(num_inchannels[i]), nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False), nn.BatchNorm2d(num_outchannels_conv3x3), nn.ReLU(True)))
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
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}


@SPPE.register_module
class PoseHighResolutionNet(nn.Module):

    def __init__(self, **cfg):
        self.inplanes = 64
        super(PoseHighResolutionNet, self).__init__()
        self._preset_cfg = cfg['PRESET']
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)
        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)
        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)
        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [(num_channels[i] * block.expansion) for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=False)
        self.final_layer = nn.Conv2d(in_channels=pre_stage_channels[0], out_channels=self._preset_cfg['NUM_JOINTS'], kernel_size=cfg['FINAL_CONV_KERNEL'], stride=1, padding=1 if cfg['FINAL_CONV_KERNEL'] == 3 else 0)
        self.pretrained_layers = cfg['PRETRAINED_LAYERS']

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False), nn.BatchNorm2d(num_channels_cur_layer[i]), nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False), nn.BatchNorm2d(outchannels), nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        modules = []
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        x = self.final_layer(y_list[0])
        return x

    def _initialize(self, pretrained=''):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers or self.pretrained_layers[0] == '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            raise ValueError('{} is not exist!'.format(pretrained))


class DUC(nn.Module):
    """
    Initialize: inplanes, planes, upscale_factor
    OUTPUT: (planes // upscale_factor^2) * ht * wd
    """

    def __init__(self, inplanes, planes, upscale_factor=2, norm_layer=nn.BatchNorm2d):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = norm_layer(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class PixelUnshuffle(nn.Module):
    """
    Initialize: inplanes, planes, upscale_factor
    OUTPUT: (planes // upscale_factor^2) * ht * wd
    """

    def __init__(self, downscale_factor=2):
        super(PixelUnshuffle, self).__init__()
        self._r = downscale_factor

    def forward(self, x):
        b, c, h, w = x.shape
        out_c = c * (self._r * self._r)
        out_h = h // self._r
        out_w = w // self._r
        x_view = x.contiguous().view(b, c, out_h, self._r, out_w, self._r)
        x_prime = x_view.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_c, out_h, out_w)
        return x_prime


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d, dcn=None):
        super(Bottleneck, self).__init__()
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=0.1)
        if self.with_dcn:
            fallback_on_stride = dcn.get('FALLBACK_ON_STRIDE', False)
            self.with_modulated_dcn = dcn.get('MODULATED', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.deformable_groups = dcn.get('DEFORM_GROUP', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(planes, self.deformable_groups * offset_channels, kernel_size=3, stride=stride, padding=1)
            self.conv2 = conv_op(planes, planes, kernel_size=3, stride=stride, padding=1, deformable_groups=self.deformable_groups, bias=False)
        self.bn2 = norm_layer(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4, momentum=0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if not self.with_dcn:
            out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = F.relu(self.bn2(self.conv2(out, offset, mask)))
        else:
            offset = self.conv2_offset(out)
            out = F.relu(self.bn2(self.conv2(out, offset)), inplace=True)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ ResNet """

    def __init__(self, architecture, norm_layer=nn.BatchNorm2d, dcn=None, stage_with_dcn=(False, False, False, False)):
        super(ResNet, self).__init__()
        self._norm_layer = norm_layer
        assert architecture in ['resnet18', 'resnet50', 'resnet101', 'resnet152']
        layers = {'resnet18': [2, 2, 2, 2], 'resnet34': [3, 4, 6, 3], 'resnet50': [3, 4, 6, 3], 'resnet101': [3, 4, 23, 3], 'resnet152': [3, 8, 36, 3]}
        self.inplanes = 64
        if architecture == 'resnet18' or architecture == 'resnet34':
            self.block = BasicBlock
        else:
            self.block = Bottleneck
        self.layers = layers[architecture]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_dcn = [(dcn if with_dcn else None) for with_dcn in stage_with_dcn]
        self.layer1 = self.make_layer(self.block, 64, self.layers[0], dcn=stage_dcn[0])
        self.layer2 = self.make_layer(self.block, 128, self.layers[1], stride=2, dcn=stage_dcn[1])
        self.layer3 = self.make_layer(self.block, 256, self.layers[2], stride=2, dcn=stage_dcn[2])
        self.layer4 = self.make_layer(self.block, 512, self.layers[3], stride=2, dcn=stage_dcn[3])

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), self._norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=self._norm_layer, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer, dcn=dcn))
        return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=False, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        if reduction:
            self.se = SELayer(planes)
        self.reduc = reduction

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.reduc:
            out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=False, norm_layer=nn.BatchNorm2d, dcn=None):
        super(Bottleneck, self).__init__()
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=0.1)
        if self.with_dcn:
            fallback_on_stride = dcn.get('FALLBACK_ON_STRIDE', False)
            self.with_modulated_dcn = dcn.get('MODULATED', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.deformable_groups = dcn.get('DEFORM_GROUP', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv2_offset = nn.Conv2d(planes, self.deformable_groups * offset_channels, kernel_size=3, stride=stride, padding=1)
            self.conv2 = conv_op(planes, planes, kernel_size=3, stride=stride, padding=1, deformable_groups=self.deformable_groups, bias=False)
        self.bn2 = norm_layer(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4, momentum=0.1)
        if reduction:
            self.se = SELayer(planes * 4)
        self.reduc = reduction
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if not self.with_dcn:
            out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        elif self.with_modulated_dcn:
            offset_mask = self.conv2_offset(out)
            offset = offset_mask[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = F.relu(self.bn2(self.conv2(out, offset, mask)))
        else:
            offset = self.conv2_offset(out)
            out = F.relu(self.bn2(self.conv2(out, offset)), inplace=True)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.reduc:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out


class SEResnet(nn.Module):
    """ SEResnet """

    def __init__(self, architecture, norm_layer=nn.BatchNorm2d, dcn=None, stage_with_dcn=(False, False, False, False)):
        super(SEResnet, self).__init__()
        self._norm_layer = norm_layer
        assert architecture in ['resnet18', 'resnet50', 'resnet101', 'resnet152']
        layers = {'resnet18': [2, 2, 2, 2], 'resnet34': [3, 4, 6, 3], 'resnet50': [3, 4, 6, 3], 'resnet101': [3, 4, 23, 3], 'resnet152': [3, 8, 36, 3]}
        self.inplanes = 64
        if architecture == 'resnet18' or architecture == 'resnet34':
            self.block = BasicBlock
        else:
            self.block = Bottleneck
        self.layers = layers[architecture]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_dcn = [(dcn if with_dcn else None) for with_dcn in stage_with_dcn]
        self.layer1 = self.make_layer(self.block, 64, self.layers[0], dcn=stage_dcn[0])
        self.layer2 = self.make_layer(self.block, 128, self.layers[1], stride=2, dcn=stage_dcn[1])
        self.layer3 = self.make_layer(self.block, 256, self.layers[2], stride=2, dcn=stage_dcn[2])
        self.layer4 = self.make_layer(self.block, 512, self.layers[3], stride=2, dcn=stage_dcn[3])

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), self._norm_layer(planes * block.expansion, momentum=0.1))
        layers = []
        if downsample is not None:
            layers.append(block(self.inplanes, planes, stride, downsample, reduction=True, norm_layer=self._norm_layer, dcn=dcn))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=self._norm_layer, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer, dcn=dcn))
        return nn.Sequential(*layers)


class SELayer(nn.Module):

    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=False, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        if reduction:
            self.se = SELayer(planes)
        self.reduc = reduction

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.reduc:
            out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=False, norm_layer=nn.BatchNorm2d, dcn=None):
        super(Bottleneck, self).__init__()
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=0.1)
        if self.with_dcn:
            fallback_on_stride = dcn.get('FALLBACK_ON_STRIDE', False)
            self.with_modulated_dcn = dcn.get('MODULATED', False)
        if stride > 1:
            conv_layers = []
            conv_layers.append(PixelUnshuffle(stride))
            if not self.with_dcn or fallback_on_stride:
                conv_layers.append(nn.Conv2d(planes * 4, planes, kernel_size=3, stride=1, padding=1, bias=False))
            else:
                conv_layers.append(DCN(planes * 4, planes, dcn, kernel_size=3, stride=1, padding=1, bias=False))
            self.conv2 = nn.Sequential(*conv_layers)
        elif not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.conv2 = DCN(planes, planes, dcn, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4, momentum=0.1)
        if reduction:
            self.se = SELayer(planes * 4)
        self.reduc = reduction
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.reduc:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out, inplace=True)
        return out


class ShuffleResnet(nn.Module):
    """ ShuffleResnet """

    def __init__(self, architecture, norm_layer=nn.BatchNorm2d, dcn=None, stage_with_dcn=(False, False, False, False)):
        super(ShuffleResnet, self).__init__()
        self._norm_layer = norm_layer
        assert architecture in ['resnet18', 'resnet50', 'resnet101', 'resnet152']
        layers = {'resnet18': [2, 2, 2, 2], 'resnet34': [3, 4, 6, 3], 'resnet50': [3, 4, 6, 3], 'resnet101': [3, 4, 23, 3], 'resnet152': [3, 8, 36, 3]}
        self.inplanes = 64
        if architecture == 'resnet18' or architecture == 'resnet34':
            self.block = BasicBlock
        else:
            self.block = Bottleneck
        self.layers = layers[architecture]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_dcn = [(dcn if with_dcn else None) for with_dcn in stage_with_dcn]
        self.layer1 = self.make_layer(self.block, 64, self.layers[0], dcn=stage_dcn[0])
        self.layer2 = self.make_layer(self.block, 128, self.layers[1], stride=2, dcn=stage_dcn[1])
        self.layer3 = self.make_layer(self.block, 256, self.layers[2], stride=2, dcn=stage_dcn[2])
        self.layer4 = self.make_layer(self.block, 512, self.layers[3], stride=2, dcn=stage_dcn[3])

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), self._norm_layer(planes * block.expansion, momentum=0.1))
        layers = []
        if downsample is not None:
            layers.append(block(self.inplanes, planes, stride, downsample, reduction=True, norm_layer=self._norm_layer, dcn=dcn))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, norm_layer=self._norm_layer, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer, dcn=dcn))
        return nn.Sequential(*layers)


class DCN(nn.Module):
    """
    Initialize: inplanes, planes, upscale_factor
    OUTPUT: (planes // upscale_factor^2) * ht * wd
    """

    def __init__(self, inplanes, planes, dcn, kernel_size, stride=1, padding=0, bias=False):
        super(DCN, self).__init__()
        fallback_on_stride = dcn.get('FALLBACK_ON_STRIDE', False)
        self.with_modulated_dcn = dcn.get('MODULATED', False)
        if fallback_on_stride:
            self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        else:
            self.deformable_groups = dcn.get('DEFORM_GROUP', 1)
            if not self.with_modulated_dcn:
                conv_op = DeformConv
                offset_channels = 18
            else:
                conv_op = ModulatedDeformConv
                offset_channels = 27
            self.conv_offset = nn.Conv2d(inplanes, self.deformable_groups * offset_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            self.conv = conv_op(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, deformable_groups=self.deformable_groups, bias=bias)

    def forward(self, x):
        if self.with_modulated_dcn:
            offset_mask = self.conv_offset(x)
            offset = offset_mask[:, :18 * self.deformable_groups, :, :]
            mask = offset_mask[:, -9 * self.deformable_groups:, :, :]
            mask = mask.sigmoid()
            out = self.conv(x, offset, mask)
        else:
            offset = self.conv_offset(x)
            out = self.conv(x, offset)
        return out


class DeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, weight, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, im2col_step=64):
        if input is not None and input.dim() != 4:
            raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        ctx.save_for_backward(input, offset, weight)
        output = input.new_empty(DeformConvFunction._output_size(input, weight, ctx.padding, ctx.dilation, ctx.stride))
        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]
        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            deform_conv_cuda.deform_conv_forward_cuda(input, weight, offset, output, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors
        grad_input = grad_offset = grad_weight = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                deform_conv_cuda.deform_conv_backward_input_cuda(input, offset, grad_output, grad_input, grad_offset, weight, ctx.bufs_[0], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                deform_conv_cuda.deform_conv_backward_parameters_cuda(input, offset, grad_output, grad_weight, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, 1, cur_im2col_step)
        return grad_input, grad_offset, grad_weight, None, None, None, None, None

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = input.size(0), channels
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += (in_size + 2 * pad - kernel) // stride_ + 1,
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError('convolution input is too small (output would be {})'.format('x'.join(map(str, output_size))))
        return output_size


deform_conv = DeformConvFunction.apply


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConv, self).__init__()
        assert not bias
        assert in_channels % groups == 0, 'in_channels {} cannot be divisible by groups {}'.format(in_channels, groups)
        assert out_channels % groups == 0, 'out_channels {} cannot be divisible by groups {}'.format(out_channels, groups)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        return deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class ModulatedDeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(ModulatedDeformConvFunction._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        deform_conv_cuda.modulated_deform_conv_cuda_forward(input, weight, bias, ctx._bufs[0], offset, mask, output, ctx._bufs[1], weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        deform_conv_cuda.modulated_deform_conv_cuda_backward(input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1], grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output, weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None
        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None, None

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding - (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding - (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


modulated_deform_conv = ModulatedDeformConvFunction.apply


class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class DeformRoIPoolingFunction(Function):

    @staticmethod
    def forward(ctx, data, rois, offset, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        out_h, out_w = _pair(out_size)
        assert isinstance(out_h, int) and isinstance(out_w, int)
        assert out_h == out_w
        out_size = out_h
        ctx.spatial_scale = spatial_scale
        ctx.out_size = out_size
        ctx.out_channels = out_channels
        ctx.no_trans = no_trans
        ctx.group_size = group_size
        ctx.part_size = out_size if part_size is None else part_size
        ctx.sample_per_part = sample_per_part
        ctx.trans_std = trans_std
        assert 0.0 <= ctx.trans_std <= 1.0
        if not data.is_cuda:
            raise NotImplementedError
        n = rois.shape[0]
        output = data.new_empty(n, out_channels, out_size, out_size)
        output_count = data.new_empty(n, out_channels, out_size, out_size)
        deform_pool_cuda.deform_psroi_pooling_cuda_forward(data, rois, offset, output, output_count, ctx.no_trans, ctx.spatial_scale, ctx.out_channels, ctx.group_size, ctx.out_size, ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        if data.requires_grad or rois.requires_grad or offset.requires_grad:
            ctx.save_for_backward(data, rois, offset)
        ctx.output_count = output_count
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        data, rois, offset = ctx.saved_tensors
        output_count = ctx.output_count
        grad_input = torch.zeros_like(data)
        grad_rois = None
        grad_offset = torch.zeros_like(offset)
        deform_pool_cuda.deform_psroi_pooling_cuda_backward(grad_output, data, rois, offset, output_count, grad_input, grad_offset, ctx.no_trans, ctx.spatial_scale, ctx.out_channels, ctx.group_size, ctx.out_size, ctx.part_size, ctx.sample_per_part, ctx.trans_std)
        return grad_input, grad_rois, grad_offset, None, None, None, None, None, None, None, None


deform_roi_pooling = DeformRoIPoolingFunction.apply


class DeformRoIPooling(nn.Module):

    def __init__(self, spatial_scale, out_size, out_channels, no_trans, group_size=1, part_size=None, sample_per_part=4, trans_std=0.0):
        super(DeformRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.out_size = _pair(out_size)
        self.out_channels = out_channels
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = out_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, data, rois, offset):
        if self.no_trans:
            offset = data.new_empty(0)
        return deform_roi_pooling(data, rois, offset, self.spatial_scale, self.out_size, self.out_channels, self.no_trans, self.group_size, self.part_size, self.sample_per_part, self.trans_std)


@SPPE.register_module
class SimplePose(nn.Module):

    def __init__(self, norm_layer=nn.BatchNorm2d, **cfg):
        super(SimplePose, self).__init__()
        self._preset_cfg = cfg['PRESET']
        self.deconv_dim = cfg['NUM_DECONV_FILTERS']
        self._norm_layer = norm_layer
        self.preact = ResNet(f"resnet{cfg['NUM_LAYERS']}")
        import torchvision.models as tm
        assert cfg['NUM_LAYERS'] in [18, 34, 50, 101, 152]
        x = eval(f"tm.resnet{cfg['NUM_LAYERS']}(pretrained=True)")
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items() if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)
        self.deconv_layers = self._make_deconv_layer()
        self.final_layer = nn.Conv2d(self.deconv_dim[2], self._preset_cfg['NUM_JOINTS'], kernel_size=1, stride=1, padding=0)

    def _make_deconv_layer(self):
        deconv_layers = []
        deconv1 = nn.ConvTranspose2d(2048, self.deconv_dim[0], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn1 = self._norm_layer(self.deconv_dim[0])
        deconv2 = nn.ConvTranspose2d(self.deconv_dim[0], self.deconv_dim[1], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn2 = self._norm_layer(self.deconv_dim[1])
        deconv3 = nn.ConvTranspose2d(self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn3 = self._norm_layer(self.deconv_dim[2])
        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*deconv_layers)

    def _initialize(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.preact(x)
        out = self.deconv_layers(out)
        out = self.final_layer(out)
        return out


class RoIAlignFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale, sample_num=0):
        out_h, out_w = _pair(out_size)
        assert isinstance(out_h, int) and isinstance(out_w, int)
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)
        output = features.new_zeros(num_rois, num_channels, out_h, out_w)
        if features.is_cuda:
            roi_align_cuda.forward(features, rois, out_h, out_w, spatial_scale, sample_num, output)
        else:
            raise NotImplementedError
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        sample_num = ctx.sample_num
        rois = ctx.saved_tensors[0]
        assert feature_size is not None and grad_output.is_cuda
        batch_size, num_channels, data_height, data_width = feature_size
        out_w = grad_output.size(3)
        out_h = grad_output.size(2)
        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = rois.new_zeros(batch_size, num_channels, data_height, data_width)
            roi_align_cuda.backward(grad_output.contiguous(), rois, out_h, out_w, spatial_scale, sample_num, grad_input)
        return grad_input, grad_rois, None, None, None


roi_align = RoIAlignFunction.apply


class RoIAlign(nn.Module):

    def __init__(self, out_size, spatial_scale=1, sample_num=0, use_torchvision=False):
        super(RoIAlign, self).__init__()
        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.use_torchvision = use_torchvision

    def forward(self, features, rois):
        if self.use_torchvision:
            from torchvision.ops import roi_align as tv_roi_align
            return tv_roi_align(features, rois, _pair(self.out_size), self.spatial_scale, self.sample_num)
        else:
            return roi_align(features, rois, self.out_size, self.spatial_scale, self.sample_num)

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(out_size={}, spatial_scale={}, sample_num={}'.format(self.out_size, self.spatial_scale, self.sample_num)
        format_str += ', use_torchvision={})'.format(self.use_torchvision)
        return format_str


def _generate_anchor_boxes(image_size, anchor_scale, anchor_configs):
    """Generates multiscale anchor boxes.

    Args:
        image_size: integer number of input image size. The input image has the same dimension for
            width and height. The image_size should be divided by the largest feature stride 2^max_level.

        anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.

        anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.

    Returns:
        anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all feature levels.

    Raises:
        ValueError: input size must be the multiple of largest feature stride.
    """
    boxes_all = []
    for _, configs in anchor_configs.items():
        boxes_level = []
        for config in configs:
            stride, octave_scale, aspect = config
            if image_size % stride != 0:
                raise ValueError('input size must be divided by the stride.')
            base_anchor_size = anchor_scale * stride * 2 ** octave_scale
            anchor_size_x_2 = base_anchor_size * aspect[0] / 2.0
            anchor_size_y_2 = base_anchor_size * aspect[1] / 2.0
            x = np.arange(stride / 2, image_size, stride)
            y = np.arange(stride / 2, image_size, stride)
            xv, yv = np.meshgrid(x, y)
            xv = xv.reshape(-1)
            yv = yv.reshape(-1)
            boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2, yv + anchor_size_y_2, xv + anchor_size_x_2))
            boxes = np.swapaxes(boxes, 0, 1)
            boxes_level.append(np.expand_dims(boxes, axis=1))
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([-1, 4]))
    anchor_boxes = np.vstack(boxes_all)
    return anchor_boxes


def _generate_anchor_configs(min_level, max_level, num_scales, aspect_ratios):
    """Generates mapping from output level to a list of anchor configurations.

    A configuration is a tuple of (num_anchors, scale, aspect_ratio).

    Args:
        min_level: integer number of minimum level of the output feature pyramid.

        max_level: integer number of maximum level of the output feature pyramid.

        num_scales: integer number representing intermediate scales added on each level.
            For instances, num_scales=2 adds two additional anchor scales [2^0, 2^0.5] on each level.

        aspect_ratios: list of tuples representing the aspect ratio anchors added on each level.
            For instances, aspect_ratios = [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.

    Returns:
        anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.
    """
    anchor_configs = {}
    for level in range(min_level, max_level + 1):
        anchor_configs[level] = []
        for scale_octave in range(num_scales):
            for aspect in aspect_ratios:
                anchor_configs[level].append((2 ** level, scale_octave / float(num_scales), aspect))
    return anchor_configs


class Anchors(nn.Module):
    """RetinaNet Anchors class."""

    def __init__(self, min_level, max_level, num_scales, aspect_ratios, anchor_scale, image_size):
        """Constructs multiscale RetinaNet anchors.

        Args:
            min_level: integer number of minimum level of the output feature pyramid.

            max_level: integer number of maximum level of the output feature pyramid.

            num_scales: integer number representing intermediate scales added
                on each level. For instances, num_scales=2 adds two additional
                anchor scales [2^0, 2^0.5] on each level.

            aspect_ratios: list of tuples representing the aspect ratio anchors added
                on each level. For instances, aspect_ratios =
                [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.

            anchor_scale: float number representing the scale of size of the base
                anchor to the feature stride 2^level.

            image_size: integer number of input image size. The input image has the
                same dimension for width and height. The image_size should be divided by
                the largest feature stride 2^max_level.
        """
        super(Anchors, self).__init__()
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.anchor_scale = anchor_scale
        self.image_size = image_size
        self.config = self._generate_configs()
        self.register_buffer('boxes', self._generate_boxes())

    def _generate_configs(self):
        """Generate configurations of anchor boxes."""
        return _generate_anchor_configs(self.min_level, self.max_level, self.num_scales, self.aspect_ratios)

    def _generate_boxes(self):
        """Generates multiscale anchor boxes."""
        boxes = _generate_anchor_boxes(self.image_size, self.anchor_scale, self.config)
        boxes = torch.from_numpy(boxes).float()
        return boxes

    def get_anchors_per_location(self):
        return self.num_scales * len(self.aspect_ratios)


class AnchorLabeler(nn.Module):
    """Labeler for multiscale anchor boxes.
    """

    def __init__(self, anchors, num_classes, match_threshold=0.5):
        """Constructs anchor labeler to assign labels to anchors.

        Args:
            anchors: an instance of class Anchors.

            num_classes: integer number representing number of classes in the dataset.

            match_threshold: float number between 0 and 1 representing the threshold
                to assign positive labels for anchors.
        """
        super(AnchorLabeler, self).__init__()
        similarity_calc = region_similarity_calculator.IouSimilarity()
        matcher = argmax_matcher.ArgMaxMatcher(match_threshold, unmatched_threshold=match_threshold, negatives_lower_than_unmatched=True, force_match_for_each_row=True)
        box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()
        self.target_assigner = target_assigner.TargetAssigner(similarity_calc, matcher, box_coder)
        self.anchors = anchors
        self.match_threshold = match_threshold
        self.num_classes = num_classes

    def _unpack_labels(self, labels):
        """Unpacks an array of labels into multiscales labels."""
        labels_unpacked = []
        anchors = self.anchors
        count = 0
        for level in range(anchors.min_level, anchors.max_level + 1):
            feat_size = int(anchors.image_size / 2 ** level)
            steps = feat_size ** 2 * anchors.get_anchors_per_location()
            indices = torch.arange(count, count + steps, device=labels.device)
            count += steps
            labels_unpacked.append(torch.index_select(labels, 0, indices).view([feat_size, feat_size, -1]))
        return labels_unpacked

    def label_anchors(self, gt_boxes, gt_labels):
        """Labels anchors with ground truth inputs.

        Args:
            gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.
                For each row, it stores [y0, x0, y1, x1] for four corners of a box.

            gt_labels: A integer tensor with shape [N, 1] representing groundtruth classes.

        Returns:
            cls_targets_dict: ordered dictionary with keys [min_level, min_level+1, ..., max_level].
                The values are tensor with shape [height_l, width_l, num_anchors]. The height_l and width_l
                represent the dimension of class logits at l-th level.

            box_targets_dict: ordered dictionary with keys [min_level, min_level+1, ..., max_level].
                The values are tensor with shape [height_l, width_l, num_anchors * 4]. The height_l and
                width_l represent the dimension of bounding box regression output at l-th level.

            num_positives: scalar tensor storing number of positives in an image.
        """
        gt_box_list = box_list.BoxList(gt_boxes)
        anchor_box_list = box_list.BoxList(self.anchors.boxes)
        cls_targets, _, box_targets, _, matches = self.target_assigner.assign(anchor_box_list, gt_box_list, gt_labels)
        cls_targets -= 1
        cls_targets = cls_targets.long()
        cls_targets_dict = self._unpack_labels(cls_targets)
        box_targets_dict = self._unpack_labels(box_targets)
        num_positives = (matches.match_results != -1).float().sum()
        return cls_targets_dict, box_targets_dict, num_positives


MAX_DETECTION_POINTS = 5000


def _post_process(config, cls_outputs, box_outputs):
    """Selects top-k predictions.

    Post-proc code adapted from Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
    and optimized for PyTorch.

    Args:
        config: a parameter dictionary that includes `min_level`, `max_level`,  `batch_size`, and `num_classes`.

        cls_outputs: an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width, num_anchors].

        box_outputs: an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width, num_anchors * 4].
    """
    batch_size = cls_outputs[0].shape[0]
    cls_outputs_all = torch.cat([cls_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, config.num_classes]) for level in range(config.num_levels)], 1)
    box_outputs_all = torch.cat([box_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, 4]) for level in range(config.num_levels)], 1)
    _, cls_topk_indices_all = torch.topk(cls_outputs_all.reshape(batch_size, -1), dim=1, k=MAX_DETECTION_POINTS)
    indices_all = cls_topk_indices_all / config.num_classes
    classes_all = cls_topk_indices_all % config.num_classes
    box_outputs_all_after_topk = torch.gather(box_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, 4))
    cls_outputs_all_after_topk = torch.gather(cls_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, config.num_classes))
    cls_outputs_all_after_topk = torch.gather(cls_outputs_all_after_topk, 2, classes_all.unsqueeze(2))
    return cls_outputs_all_after_topk, box_outputs_all_after_topk, indices_all, classes_all


def decode_box_outputs(rel_codes, anchors, output_xyxy=False):
    """Transforms relative regression coordinates to absolute positions.

    Network predictions are normalized and relative to a given anchor; this
    reverses the transformation and outputs absolute coordinates for the input image.

    Args:
        rel_codes: box regression targets.

        anchors: anchors on all feature levels.

    Returns:
        outputs: bounding boxes.

    """
    ycenter_a = (anchors[0] + anchors[2]) / 2
    xcenter_a = (anchors[1] + anchors[3]) / 2
    ha = anchors[2] - anchors[0]
    wa = anchors[3] - anchors[1]
    ty, tx, th, tw = rel_codes
    w = torch.exp(tw) * wa
    h = torch.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.0
    xmin = xcenter - w / 2.0
    ymax = ycenter + h / 2.0
    xmax = xcenter + w / 2.0
    if output_xyxy:
        out = torch.stack([xmin, ymin, xmax, ymax], dim=1)
    else:
        out = torch.stack([ymin, xmin, ymax, xmax], dim=1)
    return out


def generate_detections(cls_outputs, box_outputs, anchor_boxes, indices, classes, image_scale, nms_thres=0.5, max_dets=100):
    """Generates detections with RetinaNet model outputs and anchors.

    Args:
        cls_outputs: a torch tensor with shape [N, 1], which has the highest class
            scores on all feature levels. The N is the number of selected
            top-K total anchors on all levels.  (k being MAX_DETECTION_POINTS)

        box_outputs: a torch tensor with shape [N, 4], which stacks box regression
            outputs on all feature levels. The N is the number of selected top-k
            total anchors on all levels. (k being MAX_DETECTION_POINTS)

        anchor_boxes: a torch tensor with shape [N, 4], which stacks anchors on all
            feature levels. The N is the number of selected top-k total anchors on all levels.

        indices: a torch tensor with shape [N], which is the indices from top-k selection.

        classes: a torch tensor with shape [N], which represents the class
            prediction on all selected anchors from top-k selection.

        image_scale: a float tensor representing the scale between original image
            and input image for the detector. It is used to rescale detections for
            evaluating with the original groundtruth annotations.

    Returns:
        detections: detection results in a tensor with shape [MAX_DETECTION_POINTS, 6],
            each row representing [x, y, width, height, score, class]
    """
    anchor_boxes = anchor_boxes[(indices), :]
    boxes = decode_box_outputs(box_outputs.T.float(), anchor_boxes.T, output_xyxy=True)
    scores = cls_outputs.sigmoid().squeeze(1).float()
    human_idx = classes == 0
    boxes = boxes[human_idx]
    scores = scores[human_idx]
    classes = classes[human_idx]
    top_detection_idx = batched_nms(boxes, scores, classes, iou_threshold=nms_thres)
    top_detection_idx = top_detection_idx[:max_dets]
    boxes = boxes[top_detection_idx]
    scores = scores[top_detection_idx, None]
    classes = classes[top_detection_idx, None]
    boxes[:, (2)] -= boxes[:, (0)]
    boxes[:, (3)] -= boxes[:, (1)]
    boxes *= image_scale
    classes += 1
    detections = torch.cat([boxes, scores, classes.float()], dim=1)
    if len(top_detection_idx) < max_dets:
        detections = torch.cat([detections, torch.zeros((max_dets - len(top_detection_idx), 6), device=detections.device, dtype=detections.dtype)], dim=0)
    return detections


class DetBenchEval(nn.Module):

    def __init__(self, model, config, nms_thres=0.5, max_dets=100):
        super(DetBenchEval, self).__init__()
        self.config = config
        self.nms_thres = nms_thres
        self.max_dets = max_dets
        self.model = model
        self.anchors = Anchors(config.min_level, config.max_level, config.num_scales, config.aspect_ratios, config.anchor_scale, config.image_size)

    def forward(self, x, image_scales):
        class_out, box_out = self.model(x)
        class_out, box_out, indices, classes = _post_process(self.config, class_out, box_out)
        batch_detections = []
        for i in range(x.shape[0]):
            detections = generate_detections(class_out[i], box_out[i], self.anchors.boxes, indices[i], classes[i], image_scales[i], nms_thres=self.nms_thres, max_dets=self.max_dets)
            batch_detections.append(detections)
        return torch.stack(batch_detections, dim=0)


class DetBenchTrain(nn.Module):

    def __init__(self, model, config):
        super(DetBenchTrain, self).__init__()
        self.config = config
        self.model = model
        anchors = Anchors(config.min_level, config.max_level, config.num_scales, config.aspect_ratios, config.anchor_scale, config.image_size)
        self.anchor_labeler = AnchorLabeler(anchors, config.num_classes, match_threshold=0.5)
        self.loss_fn = None

    def forward(self, x, gt_boxes, gt_labels):
        class_out, box_out = self.model(x)
        loss = None
        gcl = []
        gbl = []
        total_positive = 0
        for i in range(x.shape[0]):
            gt_class_out, gt_box_out, num_positive = self.anchor_labeler.label_anchors(gt_boxes[i], gt_labels[i])
            gcl.append(gt_class_out)
            gbl.append(gt_box_out)
            total_positive += num_positive
        return loss


class SequentialAppend(nn.Sequential):

    def __init__(self, *args):
        super(SequentialAppend, self).__init__(*args)

    def forward(self, x: List[torch.Tensor]):
        for module in self:
            x.append(module(x))
        return x


class SequentialAppendLast(nn.Sequential):

    def __init__(self, *args):
        super(SequentialAppendLast, self).__init__(*args)

    def forward(self, x: List[torch.Tensor]):
        for module in self:
            x.append(module(x[-1]))
        return x


class ResampleFeatureMap(nn.Sequential):

    def __init__(self, in_channels, out_channels, reduction_ratio=1.0, pad_type='', pooling_type='max', norm_layer=nn.BatchNorm2d, norm_kwargs=None, conv_after_downsample=False, apply_bn=False):
        super(ResampleFeatureMap, self).__init__()
        pooling_type = pooling_type or 'max'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        self.conv_after_downsample = conv_after_downsample
        conv = None
        if in_channels != out_channels:
            conv = ConvBnAct2d(in_channels, out_channels, kernel_size=1, padding=pad_type, norm_layer=norm_layer if apply_bn else None, norm_kwargs=norm_kwargs, bias=True, act_layer=None)
        if reduction_ratio > 1:
            stride_size = int(reduction_ratio)
            if conv is not None and not self.conv_after_downsample:
                self.add_module('conv', conv)
            self.add_module('downsample', create_pool2d(pooling_type, kernel_size=stride_size + 1, stride=stride_size, padding=pad_type))
            if conv is not None and self.conv_after_downsample:
                self.add_module('conv', conv)
        else:
            if conv is not None:
                self.add_module('conv', conv)
            if reduction_ratio < 1:
                scale = int(1 // reduction_ratio)
                self.add_module('upsample', nn.UpsamplingNearest2d(scale_factor=scale))


class FpnCombine(nn.Module):

    def __init__(self, feature_info, fpn_config, fpn_channels, inputs_offsets, target_reduction, pad_type='', pooling_type='max', norm_layer=nn.BatchNorm2d, norm_kwargs=None, apply_bn_for_resampling=False, conv_after_downsample=False, weight_method='attn'):
        super(FpnCombine, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method
        self.resample = nn.ModuleDict()
        for idx, offset in enumerate(inputs_offsets):
            in_channels = fpn_channels
            if offset < len(feature_info):
                in_channels = feature_info[offset]['num_chs']
                input_reduction = feature_info[offset]['reduction']
            else:
                node_idx = offset - len(feature_info)
                input_reduction = fpn_config.nodes[node_idx]['reduction']
            reduction_ratio = target_reduction / input_reduction
            self.resample[str(offset)] = ResampleFeatureMap(in_channels, fpn_channels, reduction_ratio=reduction_ratio, pad_type=pad_type, pooling_type=pooling_type, norm_layer=norm_layer, norm_kwargs=norm_kwargs, apply_bn=apply_bn_for_resampling, conv_after_downsample=conv_after_downsample)
        if weight_method == 'attn' or weight_method == 'fastattn':
            self.edge_weights = nn.Parameter(torch.ones(len(inputs_offsets)), requires_grad=True)
        else:
            self.edge_weights = None

    def forward(self, x):
        dtype = x[0].dtype
        nodes = []
        for offset in self.inputs_offsets:
            input_node = x[offset]
            input_node = self.resample[str(offset)](input_node)
            nodes.append(input_node)
        if self.weight_method == 'attn':
            normalized_weights = torch.softmax(self.edge_weights.type(dtype), dim=0)
            x = torch.stack(nodes, dim=-1) * normalized_weights
        elif self.weight_method == 'fastattn':
            edge_weights = nn.functional.relu(self.edge_weights.type(dtype))
            weights_sum = torch.sum(edge_weights)
            x = torch.stack([(nodes[i] * edge_weights[i] / (weights_sum + 0.0001)) for i in range(len(nodes))], dim=-1)
        elif self.weight_method == 'sum':
            x = torch.stack(nodes, dim=-1)
        else:
            raise ValueError('unknown weight_method {}'.format(self.weight_method))
        x = torch.sum(x, dim=-1)
        return x


_global_config['Config'] = 4


def bifpn_sum_config(base_reduction=8):
    """BiFPN config with sum."""
    p = config.Config()
    p.nodes = [{'reduction': base_reduction << 3, 'inputs_offsets': [3, 4]}, {'reduction': base_reduction << 2, 'inputs_offsets': [2, 5]}, {'reduction': base_reduction << 1, 'inputs_offsets': [1, 6]}, {'reduction': base_reduction, 'inputs_offsets': [0, 7]}, {'reduction': base_reduction << 1, 'inputs_offsets': [1, 7, 8]}, {'reduction': base_reduction << 2, 'inputs_offsets': [2, 6, 9]}, {'reduction': base_reduction << 3, 'inputs_offsets': [3, 5, 10]}, {'reduction': base_reduction << 4, 'inputs_offsets': [4, 11]}]
    p.weight_method = 'sum'
    return p


def bifpn_attn_config():
    """BiFPN config with fast weighted sum."""
    p = bifpn_sum_config()
    p.weight_method = 'attn'
    return p


def bifpn_fa_config():
    """BiFPN config with fast weighted sum."""
    p = bifpn_sum_config()
    p.weight_method = 'fastattn'
    return p


def get_fpn_config(fpn_name):
    if not fpn_name:
        fpn_name = 'bifpn_fa'
    name_to_config = {'bifpn_sum': bifpn_sum_config(), 'bifpn_attn': bifpn_attn_config(), 'bifpn_fa': bifpn_fa_config()}
    return name_to_config[fpn_name]


class EfficientDet(nn.Module):

    def __init__(self, config, norm_kwargs=None):
        super(EfficientDet, self).__init__()
        norm_kwargs = norm_kwargs or dict(eps=0.001)
        self.backbone = create_model(config.backbone_name, features_only=True, out_indices=(2, 3, 4))
        feature_info = [dict(num_chs=f['num_chs'], reduction=f['reduction']) for i, f in enumerate(self.backbone.feature_info())]
        self.fpn = BiFpn(config, feature_info, norm_kwargs=norm_kwargs)
        self.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=norm_kwargs)
        self.box_net = HeadNet(config, num_outputs=4, norm_kwargs=norm_kwargs)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        x_class = self.class_net(x)
        x_box = self.box_net(x)
        return x_class, x_box


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Upsample(nn.Module):

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


def return_torch_unique_index(u, uv):
    n = uv.shape[1]
    first_unique = torch.zeros(n, device=u.device).long()
    for j in range(n):
        first_unique[j] = (uv[:, j:j + 1] == u).all(0).nonzero()[0]
    return first_unique


def build_targets_max(target, anchor_wh, nA, nC, nGh, nGw):
    """
    returns nT, nCorrect, tx, ty, tw, th, tconf, tcls
    """
    nB = len(target)
    txy = torch.zeros(nB, nA, nGh, nGw, 2).cuda()
    twh = torch.zeros(nB, nA, nGh, nGw, 2).cuda()
    tconf = torch.LongTensor(nB, nA, nGh, nGw).fill_(0).cuda()
    tcls = torch.ByteTensor(nB, nA, nGh, nGw, nC).fill_(0).cuda()
    tid = torch.LongTensor(nB, nA, nGh, nGw, 1).fill_(-1).cuda()
    for b in range(nB):
        t = target[b]
        t_id = t[:, (1)].clone().long().cuda()
        t = t[:, ([0, 2, 3, 4, 5])]
        nTb = len(t)
        if nTb == 0:
            continue
        gxy, gwh = t[:, 1:3].clone(), t[:, 3:5].clone()
        gxy[:, (0)] = gxy[:, (0)] * nGw
        gxy[:, (1)] = gxy[:, (1)] * nGh
        gwh[:, (0)] = gwh[:, (0)] * nGw
        gwh[:, (1)] = gwh[:, (1)] * nGh
        gi = torch.clamp(gxy[:, (0)], min=0, max=nGw - 1).long()
        gj = torch.clamp(gxy[:, (1)], min=0, max=nGh - 1).long()
        box1 = gwh
        box2 = anchor_wh.unsqueeze(1)
        inter_area = torch.min(box1, box2).prod(2)
        iou = inter_area / (box1.prod(1) + box2.prod(2) - inter_area + 1e-16)
        iou_best, a = iou.max(0)
        if nTb > 1:
            _, iou_order = torch.sort(-iou_best)
            u = torch.stack((gi, gj, a), 0)[:, (iou_order)]
            first_unique = return_torch_unique_index(u, torch.unique(u, dim=1))
            i = iou_order[first_unique]
            i = i[iou_best[i] > 0.6]
            if len(i) == 0:
                continue
            a, gj, gi, t = a[i], gj[i], gi[i], t[i]
            t_id = t_id[i]
            if len(t.shape) == 1:
                t = t.view(1, 5)
        elif iou_best < 0.6:
            continue
        tc, gxy, gwh = t[:, (0)].long(), t[:, 1:3].clone(), t[:, 3:5].clone()
        gxy[:, (0)] = gxy[:, (0)] * nGw
        gxy[:, (1)] = gxy[:, (1)] * nGh
        gwh[:, (0)] = gwh[:, (0)] * nGw
        gwh[:, (1)] = gwh[:, (1)] * nGh
        txy[b, a, gj, gi] = gxy - gxy.floor()
        twh[b, a, gj, gi] = torch.log(gwh / anchor_wh[a])
        tcls[b, a, gj, gi, tc] = 1
        tconf[b, a, gj, gi] = 1
        tid[b, a, gj, gi] = t_id.unsqueeze(1)
    tbox = torch.cat([txy, twh], -1)
    return tconf, tbox, tid


def bbox_iou(box1, box2, x1y1x2y2=False):
    """
    Returns the IoU of two bounding boxes
    """
    N, M = len(box1), len(box2)
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, (0)], box1[:, (1)], box1[:, (2)], box1[:, (3)]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, (0)], box2[:, (1)], box2[:, (2)], box2[:, (3)]
    else:
        b1_x1, b1_x2 = box1[:, (0)] - box1[:, (2)] / 2, box1[:, (0)] + box1[:, (2)] / 2
        b1_y1, b1_y2 = box1[:, (1)] - box1[:, (3)] / 2, box1[:, (1)] + box1[:, (3)] / 2
        b2_x1, b2_x2 = box2[:, (0)] - box2[:, (2)] / 2, box2[:, (0)] + box2[:, (2)] / 2
        b2_y1, b2_y2 = box2[:, (1)] - box2[:, (3)] / 2, box2[:, (1)] + box2[:, (3)] / 2
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).view(-1, 1).expand(N, M)
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).view(1, -1).expand(N, M)
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def encode_delta(gt_box_list, fg_anchor_list):
    px, py, pw, ph = fg_anchor_list[:, (0)], fg_anchor_list[:, (1)], fg_anchor_list[:, (2)], fg_anchor_list[:, (3)]
    gx, gy, gw, gh = gt_box_list[:, (0)], gt_box_list[:, (1)], gt_box_list[:, (2)], gt_box_list[:, (3)]
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    return torch.stack([dx, dy, dw, dh], dim=1)


def generate_anchor(nGh, nGw, anchor_wh):
    nA = len(anchor_wh)
    yy, xx = torch.meshgrid(torch.arange(nGh), torch.arange(nGw))
    mesh = torch.stack([xx, yy], dim=0).to(anchor_wh)
    mesh = mesh.unsqueeze(0).repeat(nA, 1, 1, 1).float()
    anchor_offset_mesh = anchor_wh.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, nGh, nGw)
    anchor_mesh = torch.cat([mesh, anchor_offset_mesh], dim=1)
    return anchor_mesh


def build_targets_thres(target, anchor_wh, nA, nC, nGh, nGw):
    ID_THRESH = 0.5
    FG_THRESH = 0.5
    BG_THRESH = 0.4
    nB = len(target)
    assert len(anchor_wh) == nA
    tbox = torch.zeros(nB, nA, nGh, nGw, 4).cuda()
    tconf = torch.LongTensor(nB, nA, nGh, nGw).fill_(0).cuda()
    tid = torch.LongTensor(nB, nA, nGh, nGw, 1).fill_(-1).cuda()
    for b in range(nB):
        t = target[b]
        t_id = t[:, (1)].clone().long().cuda()
        t = t[:, ([0, 2, 3, 4, 5])]
        nTb = len(t)
        if nTb == 0:
            continue
        gxy, gwh = t[:, 1:3].clone(), t[:, 3:5].clone()
        gxy[:, (0)] = gxy[:, (0)] * nGw
        gxy[:, (1)] = gxy[:, (1)] * nGh
        gwh[:, (0)] = gwh[:, (0)] * nGw
        gwh[:, (1)] = gwh[:, (1)] * nGh
        gxy[:, (0)] = torch.clamp(gxy[:, (0)], min=0, max=nGw - 1)
        gxy[:, (1)] = torch.clamp(gxy[:, (1)], min=0, max=nGh - 1)
        gt_boxes = torch.cat([gxy, gwh], dim=1)
        anchor_mesh = generate_anchor(nGh, nGw, anchor_wh)
        anchor_list = anchor_mesh.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        iou_pdist = bbox_iou(anchor_list, gt_boxes)
        iou_max, max_gt_index = torch.max(iou_pdist, dim=1)
        iou_map = iou_max.view(nA, nGh, nGw)
        gt_index_map = max_gt_index.view(nA, nGh, nGw)
        id_index = iou_map > ID_THRESH
        fg_index = iou_map > FG_THRESH
        bg_index = iou_map < BG_THRESH
        ign_index = (iou_map < FG_THRESH) * (iou_map > BG_THRESH)
        tconf[b][fg_index] = 1
        tconf[b][bg_index] = 0
        tconf[b][ign_index] = -1
        gt_index = gt_index_map[fg_index]
        gt_box_list = gt_boxes[gt_index]
        gt_id_list = t_id[gt_index_map[id_index]]
        if torch.sum(fg_index) > 0:
            tid[b][id_index] = gt_id_list.unsqueeze(1)
            fg_anchor_list = anchor_list.view(nA, nGh, nGw, 4)[fg_index]
            delta_target = encode_delta(gt_box_list, fg_anchor_list)
            tbox[b][fg_index] = delta_target
    return tconf, tbox, tid


def create_grids(self, img_size, nGh, nGw):
    self.stride = img_size[0] / nGw
    assert self.stride == img_size[1] / nGh
    grid_x = torch.arange(nGw).repeat((nGh, 1)).view((1, 1, nGh, nGw)).float()
    grid_y = torch.arange(nGh).repeat((nGw, 1)).transpose(0, 1).view((1, 1, nGh, nGw)).float()
    self.grid_xy = torch.stack((grid_x, grid_y), 4)
    self.anchor_vec = self.anchors / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2)


def decode_delta(delta, fg_anchor_list):
    px, py, pw, ph = fg_anchor_list[:, (0)], fg_anchor_list[:, (1)], fg_anchor_list[:, (2)], fg_anchor_list[:, (3)]
    dx, dy, dw, dh = delta[:, (0)], delta[:, (1)], delta[:, (2)], delta[:, (3)]
    gx = pw * dx + px
    gy = ph * dy + py
    gw = pw * torch.exp(dw)
    gh = ph * torch.exp(dh)
    return torch.stack([gx, gy, gw, gh], dim=1)


def decode_delta_map(delta_map, anchors):
    """
    :param: delta_map, shape (nB, nA, nGh, nGw, 4)
    :param: anchors, shape (nA,4)
    """
    nB, nA, nGh, nGw, _ = delta_map.shape
    anchor_mesh = generate_anchor(nGh, nGw, anchors)
    anchor_mesh = anchor_mesh.permute(0, 2, 3, 1).contiguous()
    anchor_mesh = anchor_mesh.unsqueeze(0).repeat(nB, 1, 1, 1, 1)
    pred_list = decode_delta(delta_map.view(-1, 4), anchor_mesh.view(-1, 4))
    pred_map = pred_list.view(nB, nA, nGh, nGw, 4)
    return pred_map


class YOLOLayer(nn.Module):

    def __init__(self, anchors, nC, nID, img_size, yolo_layer, cfg):
        super(YOLOLayer, self).__init__()
        self.layer = yolo_layer
        nA = len(anchors)
        self.anchors = torch.FloatTensor(anchors)
        self.nA = nA
        self.nC = nC
        self.nID = nID
        self.img_size = 0
        self.emb_dim = 512
        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.SoftmaxLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.s_c = nn.Parameter(-4.15 * torch.ones(1))
        self.s_r = nn.Parameter(-4.85 * torch.ones(1))
        self.s_id = nn.Parameter(-2.3 * torch.ones(1))
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)

    def forward(self, p_cat, img_size, targets=None, classifier=None, test_emb=False):
        p, p_emb = p_cat[:, :24, (...)], p_cat[:, 24:, (...)]
        nB, nGh, nGw = p.shape[0], p.shape[-2], p.shape[-1]
        if self.img_size != img_size:
            create_grids(self, img_size, nGh, nGw)
            self.grid_xy = self.grid_xy
            self.anchor_wh = self.anchor_wh
        p = p.view(nB, self.nA, self.nC + 5, nGh, nGw).permute(0, 1, 3, 4, 2).contiguous()
        p_emb = p_emb.permute(0, 2, 3, 1).contiguous()
        p_box = p[(...), :4]
        p_conf = p[(...), 4:6].permute(0, 4, 1, 2, 3)
        if targets is not None:
            if test_emb:
                tconf, tbox, tids = build_targets_max(targets, self.anchor_vec, self.nA, self.nC, nGh, nGw)
            else:
                tconf, tbox, tids = build_targets_thres(targets, self.anchor_vec, self.nA, self.nC, nGh, nGw)
            tconf, tbox, tids = tconf, tbox, tids
            mask = tconf > 0
            nT = sum([len(x) for x in targets])
            nM = mask.sum().float()
            nP = torch.ones_like(mask).sum().float()
            if nM > 0:
                lbox = self.SmoothL1Loss(p_box[mask], tbox[mask])
            else:
                FT = torch.FloatTensor if p_conf.is_cuda else torch.FloatTensor
                lbox, lconf = FT([0]), FT([0])
            lconf = self.SoftmaxLoss(p_conf, tconf)
            lid = torch.Tensor(1).fill_(0).squeeze()
            emb_mask, _ = mask.max(1)
            tids, _ = tids.max(1)
            tids = tids[emb_mask]
            embedding = p_emb[emb_mask].contiguous()
            embedding = self.emb_scale * F.normalize(embedding)
            nI = emb_mask.sum().float()
            if test_emb:
                if np.prod(embedding.shape) == 0 or np.prod(tids.shape) == 0:
                    return torch.zeros(0, self.emb_dim + 1)
                emb_and_gt = torch.cat([embedding, tids.float()], dim=1)
                return emb_and_gt
            if len(embedding) > 1:
                logits = classifier(embedding).contiguous()
                lid = self.IDLoss(logits, tids.squeeze())
            loss = torch.exp(-self.s_r) * lbox + torch.exp(-self.s_c) * lconf + torch.exp(-self.s_id) * lid + (self.s_r + self.s_c + self.s_id)
            loss *= 0.5
            return loss, loss.item(), lbox.item(), lconf.item(), lid.item(), nT
        else:
            p_conf = torch.softmax(p_conf, dim=1)[:, (1), (...)].unsqueeze(-1)
            p_emb = p_emb.unsqueeze(1).repeat(1, self.nA, 1, 1, 1).contiguous()
            p_cls = torch.zeros(nB, self.nA, nGh, nGw, 1)
            p = torch.cat([p_box, p_conf, p_cls, p_emb], dim=-1)
            p[(...), :4] = decode_delta_map(p[(...), :4], self.anchor_vec)
            p[(...), :4] *= self.stride
            return p.view(nB, -1, p.shape[-1])


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    index = 0
    prev_filters = 3
    output_filters = []
    for x in blocks:
        module = nn.Sequential()
        if x['type'] == 'net':
            continue
        if x['type'] == 'convolutional':
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module('conv_{0}'.format(index), conv)
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index), bn)
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
            if len(x['layers']) <= 2:
                try:
                    end = int(x['layers'][1])
                except:
                    end = 0
                if start > 0:
                    start = start - index
                if end > 0:
                    end = end - index
                route = EmptyLayer()
                module.add_module('route_{0}'.format(index), route)
                if end < 0:
                    filters = output_filters[index + start] + output_filters[index + end]
                else:
                    filters = output_filters[index + start]
            else:
                assert len(x['layers']) == 4
                round = EmptyLayer()
                module.add_module('route_{0}'.format(index), route)
                filters = output_filters[index + start] + output_filters[index + int(x['layers'][1])] + output_filters[index + int(x['layers'][2])] + output_filters[index + int(x['layers'][3])]
        elif x['type'] == 'shortcut':
            from_ = int(x['from'])
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut)
        elif x['type'] == 'maxpool':
            stride = int(x['stride'])
            size = int(x['size'])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)
            module.add_module('maxpool_{}'.format(index), maxpool)
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]
            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            detection = DetectionLayer(anchors)
            module.add_module('Detection_{}'.format(index), detection)
        else:
            print('Something I dunno')
            assert False
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1
    return net_info, module_list


def parse_model_cfg(path):
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

    def __init__(self, cfg_path, img_size=(1088, 608), nID=1591, test_emb=False):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_cfg(cfg_path)
        self.module_defs[0]['cfg'] = cfg_path
        self.module_defs[0]['nID'] = nID
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.loss_names = ['loss', 'box', 'conf', 'id', 'nT']
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        self.emb_dim = 512
        self.classifier = nn.Linear(self.emb_dim, nID)
        self.test_emb = test_emb

    def forward(self, x, targets=None, targets_len=None):
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        is_training = targets is not None and not self.test_emb
        layer_outputs = []
        output = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif mtype == 'yolo':
                if is_training:
                    targets = [targets[i][:int(l)] for i, l in enumerate(targets_len)]
                    x, *losses = module[0](x, self.img_size, targets, self.classifier)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                elif self.test_emb:
                    targets = [targets[i][:int(l)] for i, l in enumerate(targets_len)]
                    x = module[0](x, self.img_size, targets, self.classifier, self.test_emb)
                else:
                    x = module[0](x, self.img_size)
                output.append(x)
            layer_outputs.append(x)
        if is_training:
            self.losses['nT'] /= 3
            output = [o.squeeze() for o in output]
            return sum(output), torch.Tensor(list(self.losses.values()))
        elif self.test_emb:
            return torch.cat(output, 0)
        return torch.cat(output, 1)


class test_net(nn.Module):

    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers = num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5, 5) for x in range(num_layers)])
        self.output = nn.Linear(5, 2)

    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)


class MaxPoolStride1(nn.Module):

    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padding = int(self.pad / 2)
        padded_x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
        pooled_x = nn.MaxPool2d(self.kernel_size, 1)(padded_x)
        return pooled_x


class EmptyLayer(nn.Module):

    def __init__(self):
        super(EmptyLayer, self).__init__()


def predict_transform(prediction, inp_dim, anchors, num_classes, args):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    prediction[:, :, (0)] = torch.sigmoid(prediction[:, :, (0)])
    prediction[:, :, (1)] = torch.sigmoid(prediction[:, :, (1)])
    prediction[:, :, (4)] = torch.sigmoid(prediction[:, :, (4)])
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    if args:
        x_offset = x_offset.to(args.device)
        y_offset = y_offset.to(args.device)
    else:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    prediction[:, :, :2] += x_y_offset
    anchors = torch.FloatTensor(anchors)
    if args:
        anchors = anchors.to(args.device)
    else:
        anchors = anchors.cuda()
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors
    prediction[:, :, 5:5 + num_classes] = torch.sigmoid(prediction[:, :, 5:5 + num_classes])
    prediction[:, :, :4] *= stride
    return prediction


class DetectionLayer(nn.Module):

    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

    def forward(self, x, inp_dim, num_classes, confidence):
        x = x.data
        global args
        prediction = x
        prediction = predict_transform(prediction, inp_dim, self.anchors, num_classes, confidence, args)
        return prediction


class Upsample(nn.Module):

    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert x.data.dim() == 4
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, stride, W, stride).contiguous().view(B, C, H * stride, W * stride)
        return x


class ReOrgLayer(nn.Module):

    def __init__(self, stride=2):
        super(ReOrgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert x.data.dim() == 4
        B, C, H, W = x.data.shape
        hs = self.stride
        ws = self.stride
        assert H % hs == 0, 'The stride ' + str(self.stride) + ' is not a proper divisor of height ' + str(H)
        assert W % ws == 0, 'The stride ' + str(self.stride) + ' is not a proper divisor of height ' + str(W)
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(-2, -3).contiguous()
        x = x.view(B, C, H // hs * W // ws, hs, ws)
        x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(-1, -2).contiguous()
        x = x.view(B, C, ws * hs, H // ws, W // ws).transpose(1, 2).contiguous()
        x = x.view(B, C * ws * hs, H // ws, W // ws)
        return x


def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
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
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def get_blocks(self):
        return self.blocks

    def get_module_list(self):
        return self.module_list

    def forward(self, x, args):
        detections = []
        modules = self.blocks[1:]
        outputs = {}
        write = 0
        for i in range(len(modules)):
            module_type = modules[i]['type']
            if module_type == 'convolutional' or module_type == 'upsample' or module_type == 'maxpool':
                x = self.module_list[i](x)
                outputs[i] = x
            elif module_type == 'route':
                layers = modules[i]['layers']
                layers = [int(a) for a in layers]
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                elif len(layers) == 2:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                elif len(layers) == 4:
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    map3 = outputs[i + layers[2]]
                    map4 = outputs[i + layers[3]]
                    x = torch.cat((map1, map2, map3, map4), 1)
                outputs[i] = x
            elif module_type == 'shortcut':
                from_ = int(modules[i]['from'])
                x = outputs[i - 1] + outputs[i + from_]
                outputs[i] = x
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info['height'])
                num_classes = int(modules[i]['classes'])
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, args)
                if type(x) == int:
                    continue
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
                outputs[i] = outputs[i - 1]
        try:
            return detections
        except:
            return 0

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)
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

    def save_weights(self, savedfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks) - 1
        fp = open(savedfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header = header.numpy()
        header.tofile(fp)
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
                    cpu(bn.bias.data).numpy().tofile(fp)
                    cpu(bn.weight.data).numpy().tofile(fp)
                    cpu(bn.running_mean).numpy().tofile(fp)
                    cpu(bn.running_var).numpy().tofile(fp)
                else:
                    cpu(conv.bias.data).numpy().tofile(fp)
                cpu(conv.weight.data).numpy().tofile(fp)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DUC,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxPoolStride1,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PixelUnshuffle,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReOrgLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResampleFeatureMap,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SELayer,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SequentialAppend,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SequentialAppendLast,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Upsample,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_MVIG_SJTU_AlphaPose(_paritybench_base):
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

