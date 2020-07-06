import sys
_module = sys.modules[__name__]
del sys
SiamFC = _module
SiamFCIncep22 = _module
SiamFCNext22 = _module
SiamFCRes22 = _module
SiamRPN = _module
SiamRPNIncep22 = _module
SiamRPNPP = _module
SiamRPNRes22 = _module
SiamRPNResNeXt22 = _module
SiamRPNVGG = _module
SiamVGG = _module
dataset = _module
demo = _module
demo_rpn = _module
demo_rpn_utils = _module
demo_rpn = _module
eval_otb = _module
net = _module
run_SiamRPN = _module
test_otb = _module
utils = _module
vot = _module
vot_SiamRPN = _module
demo_utils = _module
crops = _module
parse_arguments = _module
region_to_bbox = _module
siamese = _module
siamvggtracker = _module
image = _module
label_preprocess = _module
models = _module
backbones = _module
builder = _module
heads = _module
loss = _module
lr_scheduler = _module
modules = _module
neck = _module
utils = _module
train = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import random


import torch


import numpy as np


from torch.utils.data import Dataset


import torchvision.transforms.functional as F


import time


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


from torchvision import transforms


import collections


import scipy.io


import math


from torchvision import models


import torch.nn.init as init


import torch.nn


from torch.optim.lr_scheduler import _LRScheduler


from collections import OrderedDict


from torchvision import datasets


class AlexNet(nn.Module):
    """
    AlexNet backbone
    """

    def __init__(self):
        super(AlexNet, self).__init__()
        self.feature_channel = 256
        self.feature = nn.Sequential(nn.Conv2d(3, 96, 11, 2), nn.BatchNorm2d(96, eps=1e-06, momentum=0.05), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2), nn.Conv2d(96, 256, 5, 1, groups=2), nn.BatchNorm2d(256, eps=1e-06, momentum=0.05), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2), nn.Conv2d(256, 384, 3, 1), nn.BatchNorm2d(384, eps=1e-06, momentum=0.05), nn.ReLU(inplace=True), nn.Conv2d(384, 384, 3, 1, groups=2), nn.BatchNorm2d(384, eps=1e-06, momentum=0.05), nn.ReLU(inplace=True), nn.Conv2d(384, 256, 3, 1, groups=2))

    def forward(self, x):
        x = self.feature(x)
        return x


class SiamRPN(nn.Module):

    def __init__(self):
        super(SiamRPN, self).__init__()
        self.width = int(256)
        self.height = int(256)
        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0
        self.features = AlexNet()
        self.regress_adjust = nn.Conv2d(4 * 5, 4 * 5, 1)
        self.mid()
        self._initialize_weights()

    def mid(self):
        self.conv_cls1 = nn.Conv2d(self.features.feature_channel, self.features.feature_channel * 2 * 5, kernel_size=3, stride=1, padding=0)
        self.conv_r1 = nn.Conv2d(self.features.feature_channel, self.features.feature_channel * 4 * 5, kernel_size=3, stride=1, padding=0)
        self.conv_cls2 = nn.Conv2d(self.features.feature_channel, self.features.feature_channel, kernel_size=3, stride=1, padding=0)
        self.conv_r2 = nn.Conv2d(self.features.feature_channel, self.features.feature_channel, kernel_size=3, stride=1, padding=0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def xcorr(self, z, x, channels):
        out = []
        kernel_size = z.data.size()[-1]
        for i in range(x.size(0)):
            out.append(F.conv2d(x[(i), :, :, :].unsqueeze(0), z[(i), :, :, :].unsqueeze(0).view(channels, self.features.feature_channel, kernel_size, kernel_size)))
        return torch.cat(out, dim=0)

    def forward(self, template, detection):
        template_feature = self.features(template)
        detection_feature = self.features(detection)
        kernel_score = self.conv_cls1(template_feature)
        kernel_regression = self.conv_r1(template_feature)
        conv_score = self.conv_cls2(detection_feature)
        conv_regression = self.conv_r2(detection_feature)
        pred_score = self.xcorr(kernel_score, conv_score, 10)
        pred_regression = self.regress_adjust(self.xcorr(kernel_regression, conv_regression, 20))
        return pred_score, pred_regression


class Vgg(nn.Module):
    """
    Vgg backbone
    """

    def __init__(self):
        super(Vgg, self).__init__()
        self.feature_channel = 256
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), nn.Conv2d(64, 128, kernel_size=3, stride=1), nn.ReLU(inplace=True), nn.Conv2d(128, 128, kernel_size=3, stride=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), nn.Conv2d(128, 256, kernel_size=3, stride=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, stride=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, stride=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), nn.Conv2d(256, 512, kernel_size=3, stride=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=3, stride=1), nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=3, stride=1), nn.ReLU(inplace=True), nn.Conv2d(512, 256, kernel_size=1, stride=1))

    def forward(self, x):
        x = self.features(x)
        return x


class SiamRPNVGG(SiamRPN):

    def __init__(self):
        super(SiamRPNVGG, self).__init__()
        self.features = Vgg()
        self.mid()
        self._initialize_weights()
        mod = models.vgg16(pretrained=True)
        for i in xrange(len(self.features.state_dict().items()) - 2):
            self.features.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]


def center_crop(x):
    """
    center crop layer. crop [1:-2] to eliminate padding influence.
    Crop 1 element around the tensor
    input x can be a Variable or Tensor
    """
    return x[:, :, 1:-1, 1:-1].contiguous()


class BasicBlock_C(nn.Module):
    """
    increasing cardinality is a more effective way of
    gaining accuracy than going deeper or wider
    """

    def __init__(self, in_planes, bottleneck_width=4, cardinality=32, expansion=2, last_relu=True):
        super(BasicBlock_C, self).__init__()
        inner_width = cardinality * bottleneck_width
        self.expansion = expansion
        self.basic = nn.Sequential(OrderedDict([('conv1_0', nn.Conv2d(in_planes, inner_width, 1, stride=1, bias=False)), ('bn1', nn.BatchNorm2d(inner_width)), ('act0', nn.ReLU()), ('conv3_0', nn.Conv2d(inner_width, inner_width, 3, stride=1, padding=1, groups=cardinality, bias=False)), ('bn2', nn.BatchNorm2d(inner_width)), ('act1', nn.ReLU()), ('conv1_1', nn.Conv2d(inner_width, inner_width * self.expansion, 1, stride=1, bias=False)), ('bn3', nn.BatchNorm2d(inner_width * self.expansion))]))
        self.shortcut = nn.Sequential()
        if in_planes != inner_width * self.expansion:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, inner_width * self.expansion, 1, stride=1, bias=False))
        self.bn0 = nn.BatchNorm2d(self.expansion * inner_width)
        self.last_relu = last_relu

    def forward(self, x):
        out = self.basic(x)
        out += self.shortcut(x)
        if self.last_relu:
            return center_crop(F.relu(self.bn0(out)))
        else:
            return center_crop(self.bn0(out))


class ResNeXt(nn.Module):
    """
    ResNeXt with 22 layer utilized in CVPR2019 paper.
    Usage: ResNeXt([3, 4], 32, 4)
    """

    def __init__(self, num_blocks, cardinality, bottleneck_width, expansion=2):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.expansion = expansion
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv0 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=0)
        self.bn0 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(num_blocks[0], last_relu=True)
        self.layer2 = self._make_layer(num_blocks[1], last_relu=False, stride2pool=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, num_blocks, last_relu=True, stride2pool=False):
        layers = []
        for i in range(0, num_blocks):
            if i == num_blocks - 1:
                layers.append(BasicBlock_C(self.in_planes, self.bottleneck_width, self.cardinality, self.expansion, last_relu=last_relu))
            else:
                layers.append(BasicBlock_C(self.in_planes, self.bottleneck_width, self.cardinality, self.expansion))
            self.in_planes = self.expansion * self.bottleneck_width * self.cardinality
            if i == 0 and stride2pool:
                layers.append(self.maxpool)
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        out = self.maxpool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        return out


class ResNeXt22(nn.Module):

    def __init__(self):
        super(ResNeXt22, self).__init__()
        self.features = ResNeXt(num_blocks=[3, 4], cardinality=32, bottleneck_width=4)
        self.feature_channel = 512

    def forward(self, x):
        x = self.features(x)
        return x


class SiamRPNResNeXt22(SiamRPN):

    def __init__(self):
        super(SiamRPNResNeXt22, self).__init__()
        self.features = ResNeXt22()
        self.mid()
        self._initialize_weights()


class AdjustLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = 4
            r = l + 7
            x = x[:, :, l:r, l:r]
        return x


class AdjustAllLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0])
        else:
            for i in range(self.num):
                self.add_module('downsample' + str(i + 2), AdjustLayer(in_channels[i], out_channels[i]))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample' + str(i + 2))
                out.append(adj_layer(features[i]))
            return out


def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class DepthwiseXCorr(nn.Module):

    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.conv_search = nn.Sequential(nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True), nn.Conv2d(hidden, out_channels, kernel_size=1))

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out


class RPN(nn.Module):

    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


class DepthwiseRPN(RPN):

    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(DepthwiseRPN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


class MultiRPN(RPN):

    def __init__(self, anchor_num, in_channels, weighted=False):
        super(MultiRPN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('rpn' + str(i + 2), DepthwiseRPN(anchor_num, in_channels[i], in_channels[i]))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            rpn = getattr(self, 'rpn' + str(idx))
            c, l = rpn(z_f, x_f)
            cls.append(c)
            loc.append(l)
        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s
        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 2 - stride
        if downsample is not None and dilation > 1:
            dilation = dilation // 2
            padding = dilation
        assert stride == 1 or dilation == 1, 'stride and dilation must have one equals to zero at least'
        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)
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


class ResNetPP(nn.Module):

    def __init__(self, block, layers, used_layers):
        self.inplanes = 64
        super(ResNetPP, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.feature_size = 128 * block.expansion
        self.used_layers = used_layers
        layer3 = True if 3 in used_layers else False
        layer4 = True if 4 in used_layers else False
        if layer3:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x
        if layer4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=3, stride=stride, bias=False, padding=padding, dilation=dd), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_ = self.relu(x)
        x = self.maxpool(x_)
        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        p4 = self.layer4(p3)
        out = [x_, p1, p2, p3, p4]
        out = [out[i] for i in self.used_layers]
        if len(out) == 1:
            return out[0]
        else:
            return out


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    """
    model = ResNetPP(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


class SiamRPNPP(nn.Module):

    def __init__(self):
        super(SiamRPNPP, self).__init__()
        self.features = resnet50(**{'used_layers': [2, 3, 4]})
        self.neck = AdjustAllLayer(**{'in_channels': [512, 1024, 2048], 'out_channels': [256, 256, 256]})
        self.head = MultiRPN(**{'anchor_num': 5, 'in_channels': [256, 256, 256], 'weighted': True})

    def template(self, z):
        zf = self.features(z)
        zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.features(x)
        xf = self.neck(xf)
        cls, loc = self.head(self.zf, xf)
        return {'cls': cls, 'loc': loc}

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, template, detection):
        zf = self.features(template)
        xf = self.features(detection)
        zf = self.neck(zf)
        xf = self.neck(xf)
        cls, loc = self.head(zf, xf)
        return cls, loc


class SiamRPNPPRes50(SiamRPNPP):

    def __init__(self, tracker_name='SiamRPNPP'):
        super(SiamRPNPPRes50, self).__init__(tracker_name)
        self.cfg = {'lr': 0.45, 'window_influence': 0.44, 'penalty_k': 0.04, 'instance_size': 255, 'adaptive': False}


def Image_to_Tensor(img, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]):
    zn = np.asarray(img, 'float')
    zr = zn.transpose(2, 0, 1)
    for c in range(0, 3):
        zr[c] = (zr[c] / 255 - mean[c]) / std[c]
    zt = torch.from_numpy(zr).float()
    return zt


Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])


def extract_crops_x(im, npad, pos_x, pos_y, sz_src0, sz_src1, sz_src2, sz_dst):
    c = sz_src2 / 2
    tr_x = npad + int(round(pos_x - c))
    tr_y = npad + int(round(pos_y - c))
    width = round(pos_x + c) - round(pos_x - c)
    height = round(pos_y + c) - round(pos_y - c)
    search_area = im.crop((int(tr_x), int(tr_y), int(tr_x + width), int(tr_y + height)))
    offset_s0 = (sz_src2 - sz_src0) / 2
    offset_s1 = (sz_src2 - sz_src1) / 2
    crop_s0 = search_area.crop((int(offset_s0), int(offset_s0), int(offset_s0 + round(sz_src0)), int(offset_s0 + round(sz_src0))))
    crop_s0 = crop_s0.resize((sz_dst, sz_dst), Image.BILINEAR)
    crop_s1 = search_area.crop((int(offset_s1), int(offset_s1), int(offset_s1 + round(sz_src1)), int(offset_s1 + round(sz_src1))))
    crop_s1 = crop_s1.resize((sz_dst, sz_dst), Image.BILINEAR)
    crop_s2 = search_area.resize((sz_dst, sz_dst), Image.BILINEAR)
    crop_s0 = 255.0 * F.to_tensor(crop_s0)
    crop_s1 = 255.0 * F.to_tensor(crop_s1)
    crop_s2 = 255.0 * F.to_tensor(crop_s2)
    crops = torch.stack((crop_s0, crop_s1, crop_s2))
    return crops


def extract_crops_z(im, npad, pos_x, pos_y, sz_src, sz_dst):
    c = sz_src / 2
    tr_x = npad + int(round(pos_x - c))
    tr_y = npad + int(round(pos_y - c))
    width = round(pos_x + c) - round(pos_x - c)
    height = round(pos_y + c) - round(pos_y - c)
    crop = im.crop((int(tr_x), int(tr_y), int(tr_x + width), int(tr_y + height)))
    crop = crop.resize((sz_dst, sz_dst), Image.BILINEAR)
    crops = 255.0 * F.to_tensor(crop).unsqueeze(0)
    return crops


def gen_xz(img, inbox, to='x', pdrt=1):
    box = Rectangle(inbox.x, inbox.y, inbox.width * pdrt, inbox.height * pdrt)
    x_sz = 255, 255
    z_sz = 127, 127
    bg = Image.new('RGB', (int(box.width), int(box.height)), tuple(map(int, ImageStat.Stat(img).mean)))
    bg.paste(img, (-int(box.x - 0.5 * box.width), -int(box.y - 0.5 * box.height)))
    if to == 'x':
        temp = bg.resize(x_sz)
    elif to == 'z':
        temp = bg.resize(z_sz)
    else:
        raise ValueError('Bbox format: {} was not recognized'.format(to))
    return temp


def pad_frame(im, frame_sz, pos_x, pos_y, patch_sz, avg_chan):
    c = patch_sz / 2
    xleft_pad = max(0, -int(round(pos_x - c)))
    ytop_pad = max(0, -int(round(pos_y - c)))
    xright_pad = max(0, int(round(pos_x + c)) - frame_sz[1])
    ybottom_pad = max(0, int(round(pos_y + c)) - frame_sz[0])
    npad = max((xleft_pad, ytop_pad, xright_pad, ybottom_pad))
    if avg_chan is not None:
        avg_chan = tuple([int(round(c)) for c in avg_chan])
        im_padded = ImageOps.expand(im, border=npad, fill=avg_chan)
    else:
        im_padded = ImageOps.expand(im, border=npad, fill=0)
    return im_padded, npad


class SiameseNet(nn.Module):

    def __init__(self, tracker_name):
        super(SiameseNet, self).__init__()
        self.tracker_name = tracker_name
        self.model = getattr(builder, tracker_name)()

    def forward(self, z, x):
        z = self.model.features(z)
        x = self.model.features(x)
        out = self.mdoel.head(z, x)
        return out

    def branch(self, allin):
        allout = self.model.features(allin)
        return allout

    def get_template_z(self, pos_x, pos_y, z_sz, image, design):
        if isinstance(image, six.string_types):
            image = Image.open(image).convert('RGB')
        avg_chan = ImageStat.Stat(image).mean
        frame_padded_z, npad_z = pad_frame(image, image.size, pos_x, pos_y, z_sz, avg_chan)
        z_crops = extract_crops_z(frame_padded_z, npad_z, pos_x, pos_y, z_sz, design.exemplar_sz)
        template_z = self.branch(Variable(z_crops))
        return image, template_z

    def get_template_z_new(self, pos_x, pos_y, z_sz, image, design):
        if isinstance(image, six.string_types):
            image = Image.open(image).convert('RGB')
        z = gen_xz(image, Rectangle(pos_x, pos_y, z_sz, z_sz), to='z')
        tz = Image_to_Tensor(z).unsqueeze(0)
        template_z = self.branch(Variable(tz))
        return image, template_z

    def get_scores(self, pos_x, pos_y, scaled_search_area, template_z, filename, design, final_score_sz):
        image = Image.open(filename).convert('RGB')
        avg_chan = ImageStat.Stat(image).mean
        frame_padded_x, npad_x = pad_frame(image, image.size, pos_x, pos_y, scaled_search_area[2], avg_chan)
        x_crops = extract_crops_x(frame_padded_x, npad_x, pos_x, pos_y, scaled_search_area[0], scaled_search_area[1], scaled_search_area[2], design.search_sz)
        template_x = self.branch(Variable(x_crops))
        template_z = template_z.repeat(template_x.size(0), 1, 1, 1)
        scores = self.model.head(template_z, template_x)
        scores = scores.squeeze().permute(1, 2, 0).data.cpu().numpy()
        scores_up = cv2.resize(scores, (final_score_sz, final_score_sz), interpolation=cv2.INTER_CUBIC)
        scores_up = scores_up.transpose((2, 0, 1))
        return image, scores_up

    def get_scores_new(self, pos_x, pos_y, scaled_search_area, template_z, filename, design, final_score_sz):
        image = Image.open(filename).convert('RGB')
        txs = []
        for scale in scaled_search_area:
            x = gen_xz(image, Rectangle(pos_x, pos_y, scale, scale), to='x')
            tx = Image_to_Tensor(x).unsqueeze(0)
            txs.append(tx.squeeze(0))
        x_crops = torch.stack(txs)
        template_x = self.branch(Variable(x_crops))
        template_z = template_z.repeat(template_x.size(0), 1, 1, 1)
        scores = self.model.head(template_z, template_x)
        scores = scores.squeeze().permute(1, 2, 0).data.cpu().numpy()
        scores_up = cv2.resize(scores, (final_score_sz, final_score_sz), interpolation=cv2.INTER_CUBIC)
        scores_up = scores_up.transpose((2, 0, 1))
        return image, scores_up


eps = 1e-05


class Bottleneck_CI(nn.Module):
    """
    Bottleneck with center crop layer, utilized in CVPR2019 model
    """
    expansion = 4

    def __init__(self, inplanes, planes, last_relu, stride=1, downsample=None, dilation=1):
        super(Bottleneck_CI, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 1
        if abs(dilation - 2) < eps:
            padding = 2
        if abs(dilation - 3) < eps:
            padding = 3
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

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
        if self.last_relu:
            out = self.relu(out)
        out = center_crop(out)
        return out


def center_crop7(x):
    """
    Center crop layer for stage1 of resnet. (7*7)
    input x can be a Variable or Tensor
    """
    return x[:, :, 2:-2, 2:-2].contiguous()


class ResNet(nn.Module):
    """
    ResNet with 22 layer utilized in CVPR2019 paper.
    Usage: ResNet(Bottleneck_CI, [3, 4], [True, False], [False, True], 64, [64, 128])
    """

    def __init__(self, block, layers, last_relus, s2p_flags, firstchannels=64, channels=[64, 128], dilation=1):
        self.inplanes = firstchannels
        self.stage_len = len(layers)
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, firstchannels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(firstchannels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        if s2p_flags[0]:
            self.layer1 = self._make_layer(block, channels[0], layers[0], stride2pool=True, last_relu=last_relus[0])
        else:
            self.layer1 = self._make_layer(block, channels[0], layers[0], last_relu=last_relus[0])
        if s2p_flags[1]:
            self.layer2 = self._make_layer(block, channels[1], layers[1], stride2pool=True, last_relu=last_relus[1], dilation=dilation)
        else:
            self.layer2 = self._make_layer(block, channels[1], layers[1], last_relu=last_relus[1], dilation=dilation)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, last_relu, stride=1, stride2pool=False, dilation=1):
        """
        :param block:
        :param planes:
        :param blocks:
        :param stride:
        :param stride2pool: translate (3,2) conv to (3, 1)conv + (2, 2)pool
        :return:
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, last_relu=True, stride=stride, downsample=downsample, dilation=dilation))
        if stride2pool:
            layers.append(self.maxpool)
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == blocks - 1:
                layers.append(block(self.inplanes, planes, last_relu=last_relu, dilation=dilation))
            else:
                layers.append(block(self.inplanes, planes, last_relu=True, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = center_crop7(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ResNet22(nn.Module):
    """
    FAT: fix all at first (for siamrpn)
    """

    def __init__(self):
        super(ResNet22, self).__init__()
        self.features = ResNet(Bottleneck_CI, [3, 4], [True, False], [False, True])
        self.feature_channel = 512

    def forward(self, x):
        x = self.features(x)
        return x


class Inception(nn.Module):
    """
    Inception with 22 layer utilized in CVPR2019 paper.
    Usage: Inception(InceptionM, [3, 4], [True, False])
    """

    def __init__(self, block, layers):
        self.inplanes = 64
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, 64, layers[0], pool=False)
        self.layer2 = self._make_layer(block, 320, 128, layers[1], pool=True, last_relu=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inchannels, planes, blocks, pool=True, last_relu=True):
        layers = []
        for i in range(0, blocks):
            if i == 0:
                self.inchannels = inchannels
            else:
                self.inchannels = planes * 5
            if i == 1 and pool:
                layers.append(self.maxpool)
            if i == blocks - 1 and not last_relu:
                layers.append(block(self.inchannels, planes, last_relu))
            else:
                layers.append(block(self.inchannels, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class BasicConv2d_1x1(nn.Module):
    """
    1*1 branch of inception
    """

    def __init__(self, in_channels, out_channels, last_relu=True, **kwargs):
        super(BasicConv2d_1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.last_relu = last_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.last_relu:
            return F.relu(x, inplace=True)
        else:
            return x


class BasicConv2d_3x3(nn.Module):
    """
    3*3 branch of inception
    """
    expansion = 4

    def __init__(self, inplanes, planes, last_relu=True):
        super(BasicConv2d_3x3, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.last_relu = last_relu

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.last_relu:
            out = self.relu(out)
        return out


class InceptionM(nn.Module):
    """
    Inception module with 1*1 and 3*3 branch
    """

    def __init__(self, in_channels, planes, last_relu=True):
        super(InceptionM, self).__init__()
        self.branch3x3 = BasicConv2d_3x3(in_channels, planes, last_relu)
        self.branch1x1 = BasicConv2d_1x1(in_channels, planes, last_relu, kernel_size=1)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch1x1 = self.branch1x1(x)
        outputs = [branch3x3, branch1x1]
        return center_crop(torch.cat(outputs, 1))


class Incep22(nn.Module):

    def __init__(self):
        super(Incep22, self).__init__()
        self.features = Inception(InceptionM, [3, 4])
        self.feature_channel = 640

    def forward(self, x):
        x = self.features(x)
        return x


class Bottleneck_BIG_CI(nn.Module):
    """
    Bottleneck with center crop layer, double channels in 3*3 conv layer in shortcut branch
    """
    expansion = 4

    def __init__(self, inplanes, planes, last_relu, stride=1, downsample=None, dilation=1):
        super(Bottleneck_BIG_CI, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 1
        if abs(dilation - 2) < eps:
            padding = 2
        if abs(dilation - 3) < eps:
            padding = 3
        self.conv2 = nn.Conv2d(planes, planes * 2, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = nn.Conv2d(planes * 2, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

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
        if self.last_relu:
            out = self.relu(out)
        out = center_crop(out)
        return out


class ResNet22W(nn.Module):
    """
    ResNet22W: double 3*3 layer (only) channels in residual blob
    """

    def __init__(self):
        super(ResNet22W, self).__init__()
        self.features = ResNet(Bottleneck_BIG_CI, [3, 4], [True, False], [False, True], firstchannels=64, channels=[64, 128])
        self.feature_channel = 512

    def forward(self, x):
        x = self.features(x)
        return x


class SiamFC_(nn.Module):

    def __init__(self):
        super(SiamFC_, self).__init__()
        self.features = None

    def head(self, z, x):
        n, c, h, w = x.size()
        x = x.view(1, n * c, h, w)
        out = F.conv2d(x, z, groups=n)
        out = out.view(n, 1, out.size(-2), out.size(-1))
        return out

    def feature_extractor(self, x):
        return self.features(x)

    def connector(self, template_feature, search_feature):
        pred_score = self.head(template_feature, search_feature)
        return pred_score

    def branch(self, allin):
        allout = self.feature_extractor(allin)
        return allout

    def forward(self, template, search):
        zf = self.feature_extractor(template)
        xf = self.feature_extractor(search)
        score = self.connector(zf, xf)
        return score


class SiamFC(SiamFC_):

    def __init__(self):
        super(SiamFC, self).__init__()
        self.features = AlexNet()
        self._initialize_weights()

    def forward(self, z, x):
        zf = self.features(z)
        xf = self.features(x)
        score = self.head(zf, xf)
        return score

    def head(self, z, x):
        n, c, h, w = x.size()
        x = x.view(1, n * c, h, w)
        out = F.conv2d(x, z, groups=n)
        out = out.view(n, 1, out.size(-2), out.size(-1))
        out = 0.001 * out + 0.0
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SiamVGG(nn.Module):

    def __init__(self):
        super(SiamVGG, self).__init__()
        self.features = Vgg()
        self.bn_adjust = nn.BatchNorm2d(1)
        self._initialize_weights()
        mod = models.vgg16(pretrained=True)
        for i in xrange(len(self.features.state_dict().items()) - 2):
            self.features.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]

    def forward(self, z, x):
        zf = self.features(z)
        xf = self.features(x)
        score = self.head(zf, xf)
        return score

    def head(self, z, x):
        out = []
        for i in range(x.size(0)):
            out.append(F.conv2d(x[(i), :, :, :].unsqueeze(0), z[(i), :, :, :].unsqueeze(0)))
        return self.bn_adjust(torch.cat(out, dim=0))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Corr_Up(nn.Module):
    """
    SiamFC head
    """

    def __init__(self):
        super(Corr_Up, self).__init__()

    def _conv2d_group(self, x, kernel):
        batch = x.size()[0]
        pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
        px = x.view(1, -1, x.size()[2], x.size()[3])
        po = F.conv2d(px, pk, groups=batch)
        po = po.view(batch, -1, po.size()[2], po.size()[3])
        return po

    def forward(self, z_f, x_f):
        if not self.training:
            return 0.1 * F.conv2d(x_f, z_f)
        else:
            return 0.1 * self._conv2d_group(x_f, z_f)


class SiamFCRes22(SiamFC_):

    def __init__(self, **kwargs):
        super(SiamFCRes22, self).__init__(**kwargs)
        self.features = ResNet22()
        self.head = Corr_Up()
        self.criterion = nn.BCEWithLogitsLoss()

    def _cls_loss(self, pred, label, select):
        if len(select.size()) == 0:
            return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.criterion(pred, label)

    def _weighted_BCE(self, pred, label):
        label[label == -1] = 0
        pred = pred.view(-1)
        label = label.view(-1)
        pos = Variable(label.data.eq(1).nonzero().squeeze())
        neg = Variable(label.data.eq(0).nonzero().squeeze())
        loss_pos = self._cls_loss(pred, label, pos)
        loss_neg = self._cls_loss(pred, label, neg)
        return loss_pos * 0.5 + loss_neg * 0.5

    def train_loss(self, pred, label):
        return torch.mean(self._weighted_BCE(pred, label))


class SiamFCIncep22(SiamFCRes22):

    def __init__(self, **kwargs):
        super(SiamFCIncep22, self).__init__(**kwargs)
        self.features = Incep22()


class SiamFCNext22(SiamFCRes22):

    def __init__(self, **kwargs):
        super(SiamFCNext22, self).__init__(**kwargs)
        self.features = ResNeXt22()


class SiamFCRes22W(SiamFCRes22):

    def __init__(self, **kwargs):
        super(SiamFCRes22W, self).__init__(**kwargs)
        self.features = ResNet22W()


class SiamRPNRes22(SiamRPN):

    def __init__(self):
        super(SiamRPNRes22, self).__init__()
        self.features = ResNet22()
        self.mid()
        self._initialize_weights()


class SiamRPNIncep22(SiamRPN):

    def __init__(self):
        super(SiamRPNIncep22, self).__init__()
        self.features = Incep22()
        self.mid()
        self._initialize_weights()


def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po


class UPChannelRPN(RPN):

    def __init__(self, anchor_num=5, feature_in=256):
        super(UPChannelRPN, self).__init__()
        cls_output = 2 * anchor_num
        loc_output = 4 * anchor_num
        self.template_cls_conv = nn.Conv2d(feature_in, feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in, feature_in * loc_output, kernel_size=3)
        self.search_cls_conv = nn.Conv2d(feature_in, feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in, feature_in, kernel_size=3)
        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)

    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)
        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)
        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        padding = 2 - stride
        if dilation > 1:
            padding = dilation
        dd = dilation
        pad = padding
        if downsample is not None and dilation > 1:
            dd = dilation // 2
            pad = dd
        self.conv1 = nn.Conv2d(inplanes, planes, stride=stride, dilation=dd, bias=False, kernel_size=3, padding=pad)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
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
    (AdjustAllLayer,
     lambda: ([], {'in_channels': [4, 4], 'out_channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 64, 64])], {}),
     False),
    (AdjustLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AlexNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock_C,
     lambda: ([], {'in_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv2d_1x1,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv2d_3x3,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Corr_Up,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DepthwiseRPN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64]), torch.rand([4, 256, 64, 64])], {}),
     True),
    (DepthwiseXCorr,
     lambda: ([], {'in_channels': 4, 'hidden': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Incep22,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (InceptionM,
     lambda: ([], {'in_channels': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNeXt,
     lambda: ([], {'num_blocks': [4, 4], 'cardinality': 4, 'bottleneck_width': 4}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (ResNeXt22,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (ResNet22,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (ResNet22W,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (SiamFC,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128]), torch.rand([4, 3, 128, 128])], {}),
     True),
    (SiamFCIncep22,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128]), torch.rand([4, 3, 128, 128])], {}),
     False),
    (SiamFCRes22,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128]), torch.rand([4, 3, 128, 128])], {}),
     False),
    (SiamFCRes22W,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128]), torch.rand([4, 3, 128, 128])], {}),
     False),
    (SiamRPN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128]), torch.rand([4, 3, 128, 128])], {}),
     False),
    (SiamRPNIncep22,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128]), torch.rand([4, 3, 128, 128])], {}),
     False),
    (SiamRPNPP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     False),
    (SiamRPNRes22,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128]), torch.rand([4, 3, 128, 128])], {}),
     False),
    (SiamRPNResNeXt22,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128]), torch.rand([4, 3, 128, 128])], {}),
     False),
    (UPChannelRPN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64]), torch.rand([4, 256, 64, 64])], {}),
     True),
    (Vgg,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
]

class Test_zllrunning_SiameseX_PyTorch(_paritybench_base):
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

