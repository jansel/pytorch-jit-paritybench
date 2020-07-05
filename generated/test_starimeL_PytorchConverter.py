import sys
_module = sys.modules[__name__]
del sys
FaceBoxes = _module
MobileNet = _module
resnet = _module
UNet = _module
build_face_dataset = _module
main = _module
models = _module
ConvertLayer_caffe = _module
ConvertLayer_ncnn = _module
ConvertModel = _module
ReplaceDenormals = _module
caffe_pb2 = _module
run = _module
test = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


import math


import torch.utils.model_zoo as model_zoo


import torch.nn.init as init


from torch.utils import model_zoo


from torchvision import models


import time


import random


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data


import torchvision.datasets as dset


import torchvision.transforms as transforms


import torchvision.utils as vutils


from torch.autograd import Variable


import numpy as np


from torchvision import transforms


class CReLUM(nn.Module):

    def __init__(self):
        super(CReLUM, self).__init__()

    def forward(self, x):
        return F.relu(torch.cat((x, -x), 1))


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):

    def __init__(self, in_planes, n1x1down, n1x1up, n3x3):
        super(Inception, self).__init__()
        self.conv1 = BasicConv2d(in_planes, n1x1down, kernel_size=1)
        self.pool2_1 = nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True)
        self.conv2_2 = BasicConv2d(in_planes, n1x1down, kernel_size=1)
        self.conv3_1 = BasicConv2d(in_planes, n1x1up, kernel_size=1)
        self.conv3_2 = BasicConv2d(n1x1up, n3x3, kernel_size=3, padding=1)
        self.conv4_1 = BasicConv2d(in_planes, n1x1up, kernel_size=1)
        self.conv4_2 = BasicConv2d(n1x1up, n3x3, kernel_size=3, padding=1)
        self.conv4_3 = BasicConv2d(n3x3, n3x3, kernel_size=3, padding=1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.pool2_1(x)
        y2 = self.conv2_2(y2)
        y3 = self.conv3_1(x)
        y3 = self.conv3_2(y3)
        y4 = self.conv4_1(x)
        y4 = self.conv4_2(y4)
        y4 = self.conv4_3(y4)
        return torch.cat([y1, y2, y3, y4], 1)


CRelu = CReLUM()


anchors = 21, 1, 1


class FaceBoxes(nn.Module):

    def __init__(self):
        super(FaceBoxes, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3)
        self.bn1 = nn.BatchNorm2d(16, eps=0.001)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64, eps=0.001)
        self.inception1 = Inception(128, 32, 16, 32)
        self.inception2 = Inception(128, 32, 16, 32)
        self.inception3 = Inception(128, 32, 16, 32)
        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=1, stride=1)
        self.conv3_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.score_conv1 = nn.Conv2d(128, 2 * anchors[0], kernel_size=3, stride=1, padding=1)
        self.bbox_conv1 = nn.Conv2d(128, 4 * anchors[0], kernel_size=3, stride=1, padding=1)
        self.score_conv2 = nn.Conv2d(256, 2 * anchors[1], kernel_size=3, stride=1, padding=1)
        self.bbox_conv2 = nn.Conv2d(256, 4 * anchors[1], kernel_size=3, stride=1, padding=1)
        self.score_conv3 = nn.Conv2d(256, 2 * anchors[2], kernel_size=3, stride=1, padding=1)
        self.bbox_conv3 = nn.Conv2d(256, 4 * anchors[2], kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.max_pool2d(CRelu(x), kernel_size=3, stride=2, ceil_mode=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.max_pool2d(CRelu(x), kernel_size=3, stride=2, ceil_mode=True)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        score1 = self.score_conv1(x)
        bbox1 = self.bbox_conv1(x)
        x = F.relu(self.conv3_1(x), inplace=True)
        x = F.relu(self.conv3_2(x), inplace=True)
        score2 = self.score_conv2(x)
        bbox2 = self.bbox_conv2(x)
        x = F.relu(self.conv4_1(x), inplace=True)
        x = F.relu(self.conv4_2(x), inplace=True)
        score3 = self.score_conv3(x)
        bbox3 = self.bbox_conv3(x)
        scorelist = list()
        bboxlist = list()
        scorelist.append(score1.permute(0, 2, 3, 1).contiguous())
        scorelist.append(score2.permute(0, 2, 3, 1).contiguous())
        scorelist.append(score3.permute(0, 2, 3, 1).contiguous())
        bboxlist.append(bbox1.permute(0, 2, 3, 1).contiguous())
        bboxlist.append(bbox2.permute(0, 2, 3, 1).contiguous())
        bboxlist.append(bbox3.permute(0, 2, 3, 1).contiguous())
        pscore = torch.cat([o.view(o.size(0), -1) for o in scorelist], 1)
        pbbox = torch.cat([o.view(o.size(0), -1) for o in bboxlist], 1)
        return pscore, pbbox


class MobileNet(nn.Module):

    def __init__(self):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))
        self.model = nn.Sequential(conv_bn(3, 32, 2), conv_dw(32, 64, 1), conv_dw(64, 128, 2), conv_dw(128, 128, 1), conv_dw(128, 256, 2), conv_dw(256, 256, 1), conv_dw(256, 512, 2), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 1024, 2), conv_dw(1024, 1024, 1), nn.AvgPool2d(7, ceil_mode=True))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


model_urls = {'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', 'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth', 'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', 'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth'}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = model_urls
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
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
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

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class UNetEnc(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super(UNetEnc, self).__init__()
        self.up = nn.Sequential(nn.Conv2d(in_channels, features, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True), nn.ConvTranspose2d(features, out_channels, 2, stride=2), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.up(x)


class UNetDec(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super(UNetDec, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True)]
        if dropout:
            layers += [nn.Dropout(0.5)]
        layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]
        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)


class UNet(nn.Module):

    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.dec1 = UNetDec(3, 64)
        self.dec2 = UNetDec(64, 128)
        self.dec3 = UNetDec(128, 256)
        self.dec4 = UNetDec(256, 512, dropout=True)
        self.center = nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(1024, 1024, 3, padding=1), nn.ReLU(inplace=True), nn.Dropout(), nn.ConvTranspose2d(1024, 512, 2, stride=2), nn.ReLU(inplace=True))
        self.enc4 = UNetEnc(1024, 512, 256)
        self.enc3 = UNetEnc(512, 256, 128)
        self.enc2 = UNetEnc(256, 128, 64)
        self.enc1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        center = self.center(dec4)
        enc4 = self.enc4(torch.cat([center, F.upsample_bilinear(dec4, scale_factor=center.size()[2] / dec4.size()[2])], 1))
        enc3 = self.enc3(torch.cat([enc4, F.upsample_bilinear(dec3, scale_factor=enc4.size()[2] / dec3.size()[2])], 1))
        enc2 = self.enc2(torch.cat([enc3, F.upsample_bilinear(dec2, scale_factor=enc3.size()[2] / dec2.size()[2])], 1))
        enc1 = self.enc1(torch.cat([enc2, F.upsample_bilinear(dec1, scale_factor=enc2.size()[2] / dec1.size()[2])], 1))
        return self.final(enc1)


class _netG_1(nn.Module):

    def __init__(self, ngpu, nz, nc, ngf, n_extra_layers_g):
        super(_netG_1, self).__init__()
        self.ngpu = ngpu
        main = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.LeakyReLU(0.2, inplace=True), nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.LeakyReLU(0.2, inplace=True), nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.LeakyReLU(0.2, inplace=True), nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.LeakyReLU(0.2, inplace=True))
        for t in range(n_extra_layers_g):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, ngf), nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, ngf), nn.BatchNorm2d(ngf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, ngf), nn.LeakyReLU(0.2, inplace=True))
        main.add_module('final_layer.deconv', nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
        main.add_module('final_layer.tanh', nn.Tanh())
        self.main = main

    def forward(self, input):
        return self.main(input)


class _netD_1(nn.Module):

    def __init__(self, ngpu, nz, nc, ndf, n_extra_layers_d):
        super(_netD_1, self).__init__()
        self.ngpu = ngpu
        main = nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True))
        for t in range(n_extra_layers_d):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, ndf * 8), nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, ndf * 8), nn.BatchNorm2d(ndf * 8))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, ndf * 8), nn.LeakyReLU(0.2, inplace=True))
        main.add_module('final_layers.conv', nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))
        main.add_module('final_layers.sigmoid', nn.Sigmoid())
        self.main = main

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        return output.view(-1, 1)


class _netD_2(nn.Module):

    def __init__(self, ngpu, nz, nc, ndf):
        super(_netD_2, self).__init__()
        self.ngpu = ngpu
        self.convs = nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 8, 1024, 4, 1, 0, bias=False), nn.LeakyReLU(inplace=True), nn.Dropout(0.5))
        self.fcs = nn.Sequential(nn.Linear(1024, 1024), nn.LeakyReLU(inplace=True), nn.Dropout(0.5), nn.Linear(1024, 1), nn.Sigmoid())

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        output = nn.parallel.data_parallel(self.convs, input, gpu_ids)
        output = self.fcs(output.view(-1, 1024))
        return output.view(-1, 1)


class _netG_2(nn.Module):

    def __init__(self, ngpu, nz, nc, ngf):
        super(_netG_2, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.fcs = nn.Sequential(nn.Linear(nz, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.decode_fcs = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(1024, nz))
        self.convs = nn.Sequential(nn.ConvTranspose2d(1024, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(inplace=True), nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(inplace=True), nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(inplace=True), nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(inplace=True), nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, input):
        input = self.fcs(input.view(-1, self.nz))
        gpu_ids = None
        if isinstance(input.data, torch.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        z_prediction = self.decode_fcs(input)
        input = input.view(-1, 1024, 1, 1)
        output = nn.parallel.data_parallel(self.convs, input, gpu_ids)
        return output, z_prediction


class _netG_3(nn.Module):

    def __init__(self, ngpu, nz, nc, ngf):
        super(_netG_3, self).__init__()
        self.ngpu = ngpu
        self.fcs = nn.Sequential(nn.Linear(nz, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(1024, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.convs = nn.Sequential(nn.ConvTranspose2d(1024, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(inplace=True), nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(inplace=True), nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(inplace=True), nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(inplace=True), nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, input):
        input = self.fcs(input.view(-1, nz))
        gpu_ids = None
        if isinstance(input.data, torch.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        input = input.view(-1, 1024, 1, 1)
        return nn.parallel.data_parallel(self.convs, input, gpu_ids)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CReLUM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FaceBoxes,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Inception,
     lambda: ([], {'in_planes': 4, 'n1x1down': 4, 'n1x1up': 4, 'n3x3': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (UNet,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (UNetDec,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UNetEnc,
     lambda: ([], {'in_channels': 4, 'features': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_starimeL_PytorchConverter(_paritybench_base):
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

