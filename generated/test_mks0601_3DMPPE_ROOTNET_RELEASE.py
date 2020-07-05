import sys
_module = sys.modules[__name__]
del sys
base = _module
logger = _module
resnet = _module
timer = _module
utils = _module
dir_utils = _module
pose_utils = _module
vis = _module
Human36M = _module
MPII = _module
MSCOCO = _module
MuCo = _module
MuPoTS = _module
MuPoTS_eval = _module
dataset = _module
config = _module
model = _module
test = _module
train = _module

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


import math


import time


import abc


from torch.utils.data import DataLoader


import torch.optim


import torchvision.transforms as transforms


from torch.nn.parallel.data_parallel import DataParallel


import torch


import torch.nn as nn


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import Bottleneck


from torchvision.models.resnet import model_urls


from torch.nn import functional as F


class ResNetBackbone(nn.Module):

    def __init__(self, resnet_type):
        resnet_spec = {(18): (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'), (34): (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'), (50): (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'), (101): (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'), (152): (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]
        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
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
        return x

    def init_weights(self):
        org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)
        self.load_state_dict(org_resnet)
        None


_global_config['output_shape'] = 4


class RootNet(nn.Module):

    def __init__(self):
        self.inplanes = 2048
        self.outplanes = 256
        super(RootNet, self).__init__()
        self.deconv_layers = self._make_deconv_layer(3)
        self.xy_layer = nn.Conv2d(in_channels=self.outplanes, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.depth_layer = nn.Conv2d(in_channels=self.inplanes, out_channels=1, kernel_size=1, stride=1, padding=0)

    def _make_deconv_layer(self, num_layers):
        layers = []
        inplanes = self.inplanes
        outplanes = self.outplanes
        for i in range(num_layers):
            layers.append(nn.ConvTranspose2d(in_channels=inplanes, out_channels=outplanes, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False))
            layers.append(nn.BatchNorm2d(outplanes))
            layers.append(nn.ReLU(inplace=True))
            inplanes = outplanes
        return nn.Sequential(*layers)

    def forward(self, x, k_value):
        xy = self.deconv_layers(x)
        xy = self.xy_layer(xy)
        xy = xy.view(-1, 1, cfg.output_shape[0] * cfg.output_shape[1])
        xy = F.softmax(xy, 2)
        xy = xy.view(-1, 1, cfg.output_shape[0], cfg.output_shape[1])
        hm_x = xy.sum(dim=2)
        hm_y = xy.sum(dim=3)
        coord_x = hm_x * torch.comm.broadcast(torch.arange(1, cfg.output_shape[1] + 1).type(torch.FloatTensor), devices=[hm_x.device.index])[0]
        coord_y = hm_y * torch.comm.broadcast(torch.arange(1, cfg.output_shape[0] + 1).type(torch.FloatTensor), devices=[hm_y.device.index])[0]
        coord_x = coord_x.sum(dim=2) - 1
        coord_y = coord_y.sum(dim=2) - 1
        img_feat = torch.mean(x.view(x.size(0), x.size(1), x.size(2) * x.size(3)), dim=2)
        img_feat = torch.unsqueeze(img_feat, 2)
        img_feat = torch.unsqueeze(img_feat, 3)
        gamma = self.depth_layer(img_feat)
        gamma = gamma.view(-1, 1)
        depth = gamma * k_value.view(-1, 1)
        coord = torch.cat((coord_x, coord_y, depth), dim=1)
        return coord

    def init_weights(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.xy_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        for m in self.depth_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)


class ResPoseNet(nn.Module):

    def __init__(self, backbone, root):
        super(ResPoseNet, self).__init__()
        self.backbone = backbone
        self.root = root

    def forward(self, input_img, k_value, target=None):
        fm = self.backbone(input_img)
        coord = self.root(fm, k_value)
        if target is None:
            return coord
        else:
            target_coord = target['coord']
            target_vis = target['vis']
            target_have_depth = target['have_depth']
            loss_coord = torch.abs(coord - target_coord) * target_vis
            loss_coord = (loss_coord[:, (0)] + loss_coord[:, (1)] + loss_coord[:, (2)] * target_have_depth.view(-1)) / 3.0
            return loss_coord

