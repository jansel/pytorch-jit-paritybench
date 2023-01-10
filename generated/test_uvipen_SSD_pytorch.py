import sys
_module = sys.modules[__name__]
del sys
src = _module
dataset = _module
loss = _module
model = _module
process = _module
transform = _module
utils = _module
test_dataset = _module
test_image = _module
test_video = _module
train = _module

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


from torchvision.datasets import CocoDetection


from torch.utils.data.dataloader import default_collate


import torch.nn as nn


from torchvision.models.resnet import resnet50


from torchvision.models.mobilenet import mobilenet_v2


import numpy as np


import random


import torchvision.transforms as transforms


from torchvision.ops.boxes import box_iou


import itertools


from math import sqrt


import torch.nn.functional as F


from torchvision.ops.boxes import box_convert


from torch.optim.lr_scheduler import MultiStepLR


from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter


class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """

    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0 / dboxes.scale_xy
        self.scale_wh = 1.0 / dboxes.scale_wh
        self.sl1_loss = nn.SmoothL1Loss(reduce=False)
        self.dboxes = nn.Parameter(dboxes(order='xywh').transpose(0, 1).unsqueeze(dim=0), requires_grad=False)
        self.con_loss = nn.CrossEntropyLoss(reduce=False)

    def loc_vec(self, loc):
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:]
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels

            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """
        mask = glabel > 0
        pos_num = mask.sum(dim=1)
        vec_gd = self.loc_vec(gloc)
        sl1 = self.sl1_loss(ploc, vec_gd).sum(dim=1)
        sl1 = (mask.float() * sl1).sum(dim=1)
        con = self.con_loss(plabel, glabel)
        con_neg = con.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num
        closs = (con * (mask.float() + neg_mask.float())).sum(dim=1)
        total_loss = sl1 + closs
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-06)
        ret = (total_loss * num_mask / pos_num).mean(dim=0)
        return ret


class Base(nn.Module):

    def __init__(self):
        super().__init__()

    def init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.num_classes, -1)))
        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs


class ResNet(nn.Module):

    def __init__(self):
        super().__init__()
        backbone = resnet50(pretrained=True)
        self.out_channels = [1024, 512, 512, 256, 256, 256]
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])
        conv4_block1 = self.feature_extractor[-1][0]
        conv4_block1.conv1.stride = 1, 1
        conv4_block1.conv2.stride = 1, 1
        conv4_block1.downsample[0].stride = 1, 1

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SSD(Base):

    def __init__(self, backbone=ResNet(), num_classes=81):
        super().__init__()
        self.feature_extractor = backbone
        self.num_classes = num_classes
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.loc = []
        self.conf = []
        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))
        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self.init_weights()

    def _build_additional_features(self, input_size):
        self.additional_blocks = []
        for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
            if i < 3:
                layer = nn.Sequential(nn.Conv2d(input_size, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True), nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False), nn.BatchNorm2d(output_size), nn.ReLU(inplace=True))
            else:
                layer = nn.Sequential(nn.Conv2d(input_size, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True), nn.Conv2d(channels, output_size, kernel_size=3, bias=False), nn.BatchNorm2d(output_size), nn.ReLU(inplace=True))
            self.additional_blocks.append(layer)
        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def forward(self, x):
        x = self.feature_extractor(x)
        detection_feed = [x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)
        return locs, confs


feature_maps = {}


class MobileNetV2(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_extractor = mobilenet_v2(pretrained=True).features
        self.feature_extractor[14].conv[0][2].register_forward_hook(self.get_activation())

    def get_activation(self):

        def hook(self, input, output):
            feature_maps[0] = output.detach()
        return hook

    def forward(self, x):
        x = self.feature_extractor(x)
        return feature_maps[0], x


def SeperableConv2d(in_channels, out_channels, kernel_size=3):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, padding=padding), nn.BatchNorm2d(in_channels), nn.ReLU6(), nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))


def StackedSeperableConv2d(ls_channels, multiplier):
    out_channels = 6 * multiplier
    layers = [SeperableConv2d(in_channels=in_channels, out_channels=out_channels) for in_channels in ls_channels]
    layers.append(nn.Conv2d(in_channels=ls_channels[-1], out_channels=out_channels, kernel_size=1))
    return nn.ModuleList(layers)


class SSDLite(Base):

    def __init__(self, backbone=MobileNetV2(), num_classes=81, width_mul=1.0):
        super(SSDLite, self).__init__()
        self.feature_extractor = backbone
        self.num_classes = num_classes
        self.additional_blocks = nn.ModuleList([InvertedResidual(1280, 512, stride=2, expand_ratio=0.2), InvertedResidual(512, 256, stride=2, expand_ratio=0.25), InvertedResidual(256, 256, stride=2, expand_ratio=0.5), InvertedResidual(256, 64, stride=2, expand_ratio=0.25)])
        header_channels = [round(576 * width_mul), 1280, 512, 256, 256, 64]
        self.loc = StackedSeperableConv2d(header_channels, 4)
        self.conf = StackedSeperableConv2d(header_channels, self.num_classes)
        self.init_weights()

    def forward(self, x):
        y, x = self.feature_extractor(x)
        detection_feed = [y, x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)
        return locs, confs

