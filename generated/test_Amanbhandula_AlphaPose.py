import sys
_module = sys.modules[__name__]
del sys
SPPE = _module
src = _module
main_fast_inference = _module
FastPose = _module
models = _module
hgPRM = _module
DUC = _module
PRM = _module
Residual = _module
Resnet = _module
SE_Resnet = _module
SE_module = _module
layers = _module
util_models = _module
opt = _module
utils = _module
dataset = _module
coco = _module
fuse = _module
mpii = _module
eval = _module
img = _module
pose = _module
dataloader = _module
dataloader_webcam = _module
demo = _module
fn = _module
online_demo = _module
opt = _module
pPose_nms = _module
evaluation = _module
FastPose = _module
DUC = _module
SE_Resnet = _module
SE_module = _module
coco_minival = _module
p_poseNMS = _module
train = _module
coco = _module
eval = _module
img = _module
pose = _module
video_demo = _module
webcam_demo = _module
yolo = _module
bbox = _module
cam_demo = _module
darknet = _module
detect = _module
preprocess = _module
util = _module
video_demo = _module
video_demo_half = _module

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


import torch.nn as nn


import torch.utils.data


import torch.utils.data.distributed


import torch.nn.functional as F


import numpy as np


import time


import torch._utils


from torch.autograd import Variable


from collections import defaultdict


import math


from torch import nn


from functools import reduce


import torch.utils.data as data


import scipy.misc


from torchvision import transforms


from copy import deepcopy


import matplotlib.pyplot as plt


import random


import torchvision.transforms as transforms


import torch.multiprocessing as mp


import re


import collections


from torch._six import string_classes


import copy


import pandas as pd


import itertools


class DUC(nn.Module):
    """
    Initialize: inplanes, planes, upscale_factor
    OUTPUT: (planes // upscale_factor^2) * ht * wd
    """

    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=0.1)
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
        out = F.relu(out)
        return out


class SEResnet(nn.Module):
    """ SEResnet """

    def __init__(self, architecture):
        super(SEResnet, self).__init__()
        assert architecture in ['resnet50', 'resnet101']
        self.inplanes = 64
        self.layers = [3, 4, {'resnet50': 6, 'resnet101': 23}[architecture], 3]
        self.block = Bottleneck
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(self.block, 64, self.layers[0])
        self.layer2 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4 = self.make_layer(self.block, 512, self.layers[3], stride=2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=0.1))
        layers = []
        if downsample is not None:
            layers.append(block(self.inplanes, planes, stride, downsample, reduction=True))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


parser = argparse.ArgumentParser(description='PyTorch AlphaPose Training')


opt = parser.parse_args()


class FastPose_SE(nn.Module):
    conv_dim = 128

    def __init__(self):
        super(FastPose_SE, self).__init__()
        self.preact = SEResnet('resnet101')
        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2)
        self.duc2 = DUC(256, 512, upscale_factor=2)
        self.conv_out = nn.Conv2d(self.conv_dim, opt.nClasses, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)
        out = self.conv_out(out)
        return out


def createModel():
    return FastPose_SE()


def flip(x):
    assert x.dim() == 3 or x.dim() == 4
    x = x.numpy().copy()
    if x.ndim == 3:
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    return torch.from_numpy(x.copy())


def flip_v(x, cuda=False):
    x = flip(x.cpu().data)
    if cuda:
        x = x
    x = torch.autograd.Variable(x)
    return x


def shuffleLR(x, dataset):
    flipRef = dataset.flipRef
    assert x.dim() == 3 or x.dim() == 4
    for pair in flipRef:
        dim0, dim1 = pair
        dim0 -= 1
        dim1 -= 1
        if x.dim() == 4:
            tmp = x[:, dim1].clone()
            x[:, dim1] = x[:, dim0].clone()
            x[:, dim0] = tmp.clone()
        else:
            tmp = x[dim1].clone()
            x[dim1] = x[dim0].clone()
            x[dim0] = tmp.clone()
    return x


class InferenNet(nn.Module):

    def __init__(self, kernel_size, dataset):
        super(InferenNet, self).__init__()
        model = createModel()
        None
        sys.stdout.flush()
        model.load_state_dict(torch.load('./models/sppe/duc_se.pth'))
        model.eval()
        self.pyranet = model
        self.dataset = dataset

    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, 17)
        flip_out = self.pyranet(flip_v(x))
        flip_out = flip_out.narrow(1, 0, 17)
        flip_out = flip_v(shuffleLR(flip_out, self.dataset))
        out = (flip_out + out) / 2
        return out


class InferenNet_fast(nn.Module):

    def __init__(self, kernel_size, dataset):
        super(InferenNet_fast, self).__init__()
        model = createModel()
        None
        model.load_state_dict(torch.load('./models/sppe/duc_se.pth'))
        model.eval()
        self.pyranet = model
        self.dataset = dataset

    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, 17)
        return out


class FastPose(nn.Module):
    DIM = 128

    def __init__(self):
        super(FastPose, self).__init__()
        self.preact = SEResnet('resnet101')
        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2)
        self.duc2 = DUC(256, 512, upscale_factor=2)
        self.conv_out = nn.Conv2d(self.DIM, opt.nClasses, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Variable):
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)
        out = self.conv_out(out)
        return out


class CaddTable(nn.Module):

    def __init__(self, inplace=False):
        super(CaddTable, self).__init__()
        self.inplace = inplace

    def forward(self, x: (Variable or list)):
        return torch.stack(x, 0).sum(0)


class ConcatTable(nn.Module):

    def __init__(self, module_list=None):
        super(ConcatTable, self).__init__()
        self.modules_list = nn.ModuleList(module_list)

    def forward(self, x: Variable):
        y = []
        for i in range(len(self.modules_list)):
            y.append(self.modules_list[i](x))
        return y

    def add(self, module):
        self.modules_list.append(module)


def convBlock(numIn, numOut, stride, net_type):
    s_list = []
    if net_type != 'no_preact':
        s_list.append(nn.BatchNorm2d(numIn))
        s_list.append(nn.ReLU(True))
    conv1 = nn.Conv2d(numIn, numOut // 2, kernel_size=1)
    if opt.init:
        nn.init.xavier_normal(conv1.weight, gain=math.sqrt(1 / 2))
    s_list.append(conv1)
    s_list.append(nn.BatchNorm2d(numOut // 2))
    s_list.append(nn.ReLU(True))
    conv2 = nn.Conv2d(numOut // 2, numOut // 2, kernel_size=3, stride=stride, padding=1)
    if opt.init:
        nn.init.xavier_normal(conv2.weight)
    s_list.append(conv2)
    s_list.append(nn.BatchNorm2d(numOut // 2))
    s_list.append(nn.ReLU(True))
    conv3 = nn.Conv2d(numOut // 2, numOut, kernel_size=1)
    if opt.init:
        nn.init.xavier_normal(conv3.weight)
    s_list.append(conv3)
    return nn.Sequential(*s_list)


class Identity(nn.Module):

    def __init__(self, params=None):
        super(Identity, self).__init__()
        self.params = nn.ParameterList(params)

    def forward(self, x: (Variable or list)):
        return x


def skipLayer(numIn, numOut, stride, useConv):
    if numIn == numOut and stride == 1 and not useConv:
        return Identity()
    else:
        conv1 = nn.Conv2d(numIn, numOut, kernel_size=1, stride=stride)
        if opt.init:
            nn.init.xavier_normal(conv1.weight, gain=math.sqrt(1 / 2))
        return nn.Sequential(nn.BatchNorm2d(numIn), nn.ReLU(True), conv1)


def Residual(numIn, numOut, *arg, stride=1, net_type='preact', useConv=False, **kw):
    con = ConcatTable([convBlock(numIn, numOut, stride, net_type), skipLayer(numIn, numOut, stride, useConv)])
    cadd = CaddTable(True)
    return nn.Sequential(con, cadd)


class Hourglass(nn.Module):

    def __init__(self, n, nFeats, nModules, inputResH, inputResW, net_type, B, C):
        super(Hourglass, self).__init__()
        self.ResidualUp = ResidualPyramid if n >= 2 else Residual
        self.ResidualDown = ResidualPyramid if n >= 3 else Residual
        self.depth = n
        self.nModules = nModules
        self.nFeats = nFeats
        self.net_type = net_type
        self.B = B
        self.C = C
        self.inputResH = inputResH
        self.inputResW = inputResW
        up1 = self._make_residual(self.ResidualUp, False, inputResH, inputResW)
        low1 = nn.Sequential(nn.MaxPool2d(2), self._make_residual(self.ResidualDown, False, inputResH / 2, inputResW / 2))
        if n > 1:
            low2 = Hourglass(n - 1, nFeats, nModules, inputResH / 2, inputResW / 2, net_type, B, C)
        else:
            low2 = self._make_residual(self.ResidualDown, False, inputResH / 2, inputResW / 2)
        low3 = self._make_residual(self.ResidualDown, True, inputResH / 2, inputResW / 2)
        up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upperBranch = up1
        self.lowerBranch = nn.Sequential(low1, low2, low3, up2)

    def _make_residual(self, resBlock, useConv, inputResH, inputResW):
        layer_list = []
        for i in range(self.nModules):
            layer_list.append(resBlock(self.nFeats, self.nFeats, inputResH, inputResW, stride=1, net_type=self.net_type, useConv=useConv, baseWidth=self.B, cardinality=self.C))
        return nn.Sequential(*layer_list)

    def forward(self, x: Variable):
        up1 = self.upperBranch(x)
        up2 = self.lowerBranch(x)
        out = torch.add(up1, up2)
        return out


class PyraNet(nn.Module):

    def __init__(self):
        super(PyraNet, self).__init__()
        B, C = opt.baseWidth, opt.cardinality
        self.inputResH = opt.inputResH / 4
        self.inputResW = opt.inputResW / 4
        self.nStack = opt.nStack
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        if opt.init:
            nn.init.xavier_normal(conv1.weight, gain=math.sqrt(1 / 3))
        cnv1 = nn.Sequential(conv1, nn.BatchNorm2d(64), nn.ReLU(True))
        r1 = nn.Sequential(ResidualPyramid(64, 128, opt.inputResH / 2, opt.inputResW / 2, stride=1, net_type='no_preact', useConv=False, baseWidth=B, cardinality=C), nn.MaxPool2d(2))
        r4 = ResidualPyramid(128, 128, self.inputResH, self.inputResW, stride=1, net_type='preact', useConv=False, baseWidth=B, cardinality=C)
        r5 = ResidualPyramid(128, opt.nFeats, self.inputResH, self.inputResW, stride=1, net_type='preact', useConv=False, baseWidth=B, cardinality=C)
        self.preact = nn.Sequential(cnv1, r1, r4, r5)
        self.stack_lin = nn.ModuleList()
        self.stack_out = nn.ModuleList()
        self.stack_lin_ = nn.ModuleList()
        self.stack_out_ = nn.ModuleList()
        for i in range(self.nStack):
            hg = Hourglass(4, opt.nFeats, opt.nResidual, self.inputResH, self.inputResW, 'preact', B, C)
            conv1 = nn.Conv2d(opt.nFeats, opt.nFeats, kernel_size=1, stride=1, padding=0)
            if opt.init:
                nn.init.xavier_normal(conv1.weight, gain=math.sqrt(1 / 2))
            lin = nn.Sequential(hg, nn.BatchNorm2d(opt.nFeats), nn.ReLU(True), conv1, nn.BatchNorm2d(opt.nFeats), nn.ReLU(True))
            tmpOut = nn.Conv2d(opt.nFeats, opt.nClasses, kernel_size=1, stride=1, padding=0)
            if opt.init:
                nn.init.xavier_normal(tmpOut.weight)
            self.stack_lin.append(lin)
            self.stack_out.append(tmpOut)
            if i < self.nStack - 1:
                lin_ = nn.Conv2d(opt.nFeats, opt.nFeats, kernel_size=1, stride=1, padding=0)
                tmpOut_ = nn.Conv2d(opt.nClasses, opt.nFeats, kernel_size=1, stride=1, padding=0)
                if opt.init:
                    nn.init.xavier_normal(lin_.weight)
                    nn.init.xavier_normal(tmpOut_.weight)
                self.stack_lin_.append(lin_)
                self.stack_out_.append(tmpOut_)

    def forward(self, x: Variable):
        out = []
        inter = self.preact(x)
        for i in range(self.nStack):
            lin = self.stack_lin[i](inter)
            tmpOut = self.stack_out[i](lin)
            out.append(tmpOut)
            if i < self.nStack - 1:
                lin_ = self.stack_lin_[i](lin)
                tmpOut_ = self.stack_out_[i](tmpOut)
                inter = inter + lin_ + tmpOut_
        return out


class PyraNet_Inference(nn.Module):

    def __init__(self):
        super(PyraNet_Inference, self).__init__()
        B, C = opt.baseWidth, opt.cardinality
        self.inputResH = opt.inputResH / 4
        self.inputResW = opt.inputResW / 4
        self.nStack = opt.nStack
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        if opt.init:
            nn.init.xavier_normal(conv1.weight, gain=math.sqrt(1 / 3))
        cnv1 = nn.Sequential(conv1, nn.BatchNorm2d(64), nn.ReLU(True))
        r1 = nn.Sequential(ResidualPyramid(64, 128, opt.inputResH / 2, opt.inputResW / 2, stride=1, net_type='no_preact', useConv=False, baseWidth=B, cardinality=C), nn.MaxPool2d(2))
        r4 = ResidualPyramid(128, 128, self.inputResH, self.inputResW, stride=1, net_type='preact', useConv=False, baseWidth=B, cardinality=C)
        r5 = ResidualPyramid(128, opt.nFeats, self.inputResH, self.inputResW, stride=1, net_type='preact', useConv=False, baseWidth=B, cardinality=C)
        self.preact = nn.Sequential(cnv1, r1, r4, r5)
        self.stack_lin = nn.ModuleList()
        self.stack_out = nn.ModuleList()
        self.stack_lin_ = nn.ModuleList()
        self.stack_out_ = nn.ModuleList()
        for i in range(self.nStack):
            hg = Hourglass(4, opt.nFeats, opt.nResidual, self.inputResH, self.inputResW, 'preact', B, C)
            conv1 = nn.Conv2d(opt.nFeats, opt.nFeats, kernel_size=1, stride=1, padding=0)
            if opt.init:
                nn.init.xavier_normal(conv1.weight, gain=math.sqrt(1 / 2))
            lin = nn.Sequential(hg, nn.BatchNorm2d(opt.nFeats), nn.ReLU(True), conv1, nn.BatchNorm2d(opt.nFeats), nn.ReLU(True))
            tmpOut = nn.Conv2d(opt.nFeats, opt.nClasses, kernel_size=1, stride=1, padding=0)
            if opt.init:
                nn.init.xavier_normal(tmpOut.weight)
            self.stack_lin.append(lin)
            self.stack_out.append(tmpOut)
            if i < self.nStack - 1:
                lin_ = nn.Conv2d(opt.nFeats, opt.nFeats, kernel_size=1, stride=1, padding=0)
                tmpOut_ = nn.Conv2d(opt.nClasses, opt.nFeats, kernel_size=1, stride=1, padding=0)
                if opt.init:
                    nn.init.xavier_normal(lin_.weight)
                    nn.init.xavier_normal(tmpOut_.weight)
                self.stack_lin_.append(lin_)
                self.stack_out_.append(tmpOut_)

    def forward(self, x: Variable):
        inter = self.preact(x)
        for i in range(self.nStack):
            lin = self.stack_lin[i](inter)
            tmpOut = self.stack_out[i](lin)
            out = tmpOut
            if i < self.nStack - 1:
                lin_ = self.stack_lin_[i](lin)
                tmpOut_ = self.stack_out_[i](tmpOut)
                inter = inter + lin_ + tmpOut_
        return out


class ResNet(nn.Module):
    """ Resnet """

    def __init__(self, architecture):
        super(ResNet, self).__init__()
        assert architecture in ['resnet50', 'resnet101']
        self.inplanes = 64
        self.layers = [3, 4, {'resnet50': 6, 'resnet101': 23}[architecture], 3]
        self.block = Bottleneck
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.layer1 = self.make_layer(self.block, 64, self.layers[0])
        self.layer2 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4 = self.make_layer(self.block, 512, self.layers[3], stride=2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


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


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    if CUDA:
        x_offset = x_offset
        y_offset = y_offset
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    prediction[:, :, :2] += x_y_offset
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors
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
        global CUDA
        prediction = x
        prediction = predict_transform(prediction, inp_dim, self.anchors, num_classes, confidence, CUDA)
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
            None
            assert False
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1
    return net_info, module_list


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

    def forward(self, x, CUDA):
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
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
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
    (ConcatTable,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DUC,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MaxPoolStride1,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReOrgLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNet,
     lambda: ([], {'architecture': 'resnet50'}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SELayer,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SEResnet,
     lambda: ([], {'architecture': 'resnet50'}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Upsample,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Amanbhandula_AlphaPose(_paritybench_base):
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

