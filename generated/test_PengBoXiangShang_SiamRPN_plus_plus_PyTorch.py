import sys
_module = sys.modules[__name__]
del sys
config = _module
dataloader = _module
dataset = _module
illustration = _module
RPN = _module
SiamRPN = _module
network = _module
customized_resnet = _module
preprocessing = _module
create_dataset = _module
create_lmdb = _module
train = _module
AverageMeter = _module
Logger = _module
utils = _module
generate_anchors = _module
loss = _module
tools = _module

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


import matplotlib.pyplot as plt


from torch.utils.data.dataset import Dataset


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.modules.module import Module


import torch.utils.model_zoo as model_zoo


import collections


import time


import torch.backends.cudnn as cudnn


import torchvision


import torchvision.transforms as transforms


from torch.autograd import Variable


from torch.optim import lr_scheduler


from torch.utils.data import DataLoader


import random


class RPN(nn.Module):

    def __init__(self):
        super(RPN, self).__init__()
        self.adj_1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256))
        self.adj_2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256))
        self.adj_3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256))
        self.adj_4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256))
        self.fusion_module_1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.fusion_module_2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.Box_Head = nn.Sequential(nn.Conv2d(256, 4 * 5, kernel_size=1, padding=0, stride=1))
        self.Cls_Head = nn.Sequential(nn.Conv2d(256, 2 * 5, kernel_size=1, padding=0, stride=1))

    def forward(self, examplar_feature_map, search_region_feature_map, BATCH_SIZE):
        cropped_examplar_feature_map = F.pad(examplar_feature_map, (-4, -4, -4, -4))
        adj_1_output = self.adj_1(cropped_examplar_feature_map)
        adj_1_output = adj_1_output.reshape(-1, 7, 7)
        adj_1_output = adj_1_output.unsqueeze(0).permute(1, 0, 2, 3)
        adj_2_output = self.adj_2(search_region_feature_map)
        adj_2_output = adj_2_output.reshape(-1, 31, 31)
        adj_2_output = adj_2_output.unsqueeze(0)
        adj_3_output = self.adj_3(cropped_examplar_feature_map)
        adj_3_output = adj_3_output.reshape(-1, 7, 7)
        adj_3_output = adj_3_output.unsqueeze(0).permute(1, 0, 2, 3)
        adj_4_output = self.adj_4(search_region_feature_map)
        adj_4_output = adj_4_output.reshape(-1, 31, 31)
        adj_4_output = adj_4_output.unsqueeze(0)
        depthwise_cross_reg = F.conv2d(adj_2_output, adj_1_output, bias=None, stride=1, padding=0, groups=adj_1_output.size()[0]).squeeze()
        depthwise_cross_cls = F.conv2d(adj_4_output, adj_3_output, bias=None, stride=1, padding=0, groups=adj_3_output.size()[0]).squeeze()
        depthwise_cross_reg = depthwise_cross_reg.reshape(-1, 256, 25, 25)
        depthwise_cross_cls = depthwise_cross_cls.reshape(-1, 256, 25, 25)
        depthwise_cross_reg = self.fusion_module_1(depthwise_cross_reg)
        depthwise_cross_cls = self.fusion_module_2(depthwise_cross_cls)
        bbox_regression_prediction = self.Box_Head(depthwise_cross_reg)
        cls_prediction = self.Cls_Head(depthwise_cross_cls)
        return cls_prediction, bbox_regression_prediction


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)


def conv3x3_with_dilation(in_planes, out_planes, stride=1, padding=2, dilation_ratio=2, groups=1):
    """3x3 convolution with padding and dilation"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation_ratio, groups=groups, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, padding=1, dilation_ratio=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        if padding == 1 and dilation_ratio == 1:
            self.conv2 = conv3x3(planes, planes, stride, groups)
        else:
            self.conv2 = conv3x3_with_dilation(planes, planes, stride, padding, dilation_ratio, groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
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
        out = self.relu(out)
        return out


downsampling_config_dict = {'layer1': {'ksize': 1, 'padding': 0}, 'layer2': {'ksize': 3, 'padding': 0}, 'layer3': {'ksize': 1, 'padding': 0}, 'layer4': {'ksize': 1, 'padding': 0}}


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]
        self.inplanes = planes[0]
        self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, planes[0], layers[0], groups=groups, norm_layer=norm_layer, padding=1, dilation_ratio=1, layer_name='layer1')
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer, padding=0, dilation_ratio=1, layer_name='layer2')
        self.extra_1x1_conv3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=1, groups=groups, norm_layer=norm_layer, padding=2, dilation_ratio=2, layer_name='layer3')
        self.extra_1x1_conv4 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=1, groups=groups, norm_layer=norm_layer, padding=4, dilation_ratio=4, layer_name='layer4')
        self.extra_1x1_conv5 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[3] * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None, padding=1, dilation_ratio=1, layer_name=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layer_downsampling_config_dict = downsampling_config_dict[layer_name]
            ksize = layer_downsampling_config_dict['ksize']
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, stride=stride, kernel_size=ksize), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, norm_layer, padding, dilation_ratio))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        conv_3_output = self.extra_1x1_conv3(x)
        x = self.layer3(x)
        conv_4_output = self.extra_1x1_conv4(x)
        x = self.layer4(x)
        conv_5_output = self.extra_1x1_conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, conv_3_output, conv_4_output, conv_5_output


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


class SiamRPN(nn.Module):

    def __init__(self):
        super(SiamRPN, self).__init__()
        self.examplar_branch = resnet50()
        self.search_region_branch = resnet50()
        self.conv3_3_RPN = RPN()
        self.conv4_6_RPN = RPN()
        self.conv5_3_RPN = RPN()
        self.weighted_sum_layer_alpha = nn.Conv2d(30, 10, kernel_size=1, padding=0, groups=10)
        self.weighted_sum_layer_beta = nn.Conv2d(60, 20, kernel_size=1, padding=0, groups=20)

    def forward(self, examplar, search_region):
        _, examplar_conv_3_output, examplar_conv_4_output, examplar_conv_5_output = self.examplar_branch(examplar)
        _, search_region_conv_3_output, search_region_conv_4_output, search_region_conv_5_output = self.search_region_branch(search_region)
        conv3_3_cls_prediction, conv3_3_bbox_regression_prediction = self.conv3_3_RPN(examplar_conv_3_output, search_region_conv_3_output, examplar.size()[0])
        conv4_6_cls_prediction, conv4_6_bbox_regression_prediction = self.conv4_6_RPN(examplar_conv_4_output, search_region_conv_4_output, examplar.size()[0])
        conv5_3_cls_prediction, conv5_3_bbox_regression_prediction = self.conv5_3_RPN(examplar_conv_5_output, search_region_conv_5_output, examplar.size()[0])
        stacked_cls_prediction = torch.cat((conv3_3_cls_prediction, conv4_6_cls_prediction, conv5_3_cls_prediction), 2).reshape(examplar.size()[0], 10, -1, 25, 25).reshape(examplar.size()[0], -1, 25, 25)
        stacked_regression_prediction = torch.cat((conv3_3_bbox_regression_prediction, conv4_6_bbox_regression_prediction, conv5_3_bbox_regression_prediction), 2).reshape(examplar.size()[0], 20, -1, 25, 25).reshape(examplar.size()[0], -1, 25, 25)
        fused_cls_prediction = self.weighted_sum_layer_alpha(stacked_cls_prediction)
        fused_regression_prediction = self.weighted_sum_layer_beta(stacked_regression_prediction)
        return fused_cls_prediction, fused_regression_prediction

