import sys
_module = sys.modules[__name__]
del sys
affineFace = _module
calcAffine = _module
parts2lms = _module
points2heatmap = _module
test = _module
warper = _module
loader = _module
dataset_basic = _module
dataset_loader_demo = _module
dataset_loader_train = _module
base_model = _module
spade_model = _module
ResNet = _module
appear_decoder_net = _module
appear_encoder_net = _module
base_net = _module
discriminator_net = _module
face_id_mlp_net = _module
face_id_net = _module
generaotr_net = _module
generator_net_concat_1Layer = _module
vgg_net = _module
opt = _module
config = _module
configTrain = _module
test = _module
utils = _module
affine_util = _module
metric = _module
transforms = _module

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


import numpy as np


import torch


import torch.utils.data


import random


import copy


import torch as th


from collections import OrderedDict


import logging


import torch.nn.functional as F


import itertools


import torch.nn as nn


import math


import torch.utils.model_zoo as model_zoo


import torch.nn.utils.weight_norm as weight_norm


from torch import nn


from torch.nn import init


import functools


from torch.optim import lr_scheduler


from torch.nn import Parameter


from torchvision.models import vgg16


import time


import scipy.misc as m


import torchvision.utils as vutils


import inspect


import re


import collections


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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


class SEBlock(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction), nn.PReLU(), nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.prelu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class ResNetFace(nn.Module):

    def __init__(self, block, layers, input_nc, use_se=True):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNetFace, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc5 = nn.Linear(512 * 13 * 13, 512)
        self.bn5 = nn.BatchNorm1d(512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.bn5(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc5 = nn.Linear(512 * 8 * 8, 512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        return x


class appearDec(nn.Module):

    def __init__(self, input_c, norm_layer, size_=256):
        super(appearDec, self).__init__()
        layers = []
        channel_list = [1024, 1024, 1024, 1024]
        c0 = 1024
        for cc in channel_list:
            layers.append(nn.ConvTranspose2d(c0, cc, 4, 2, 1))
            layers.append(norm_layer(cc))
            layers.append(nn.ReLU(True))
            c0 = cc
        self.decoder16 = nn.Sequential(*layers)
        self.decoder32 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 2, 1), norm_layer(512), nn.ReLU(True))
        self.decoder64 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), norm_layer(256), nn.ReLU(True))
        self.decoder128 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), norm_layer(128), nn.ReLU(True))
        layers = []
        layers.append(nn.ConvTranspose2d(128, 3, 4, 2, 1))
        layers.append(nn.Tanh())
        self.decoder256 = nn.Sequential(*layers)

    def forward(self, input):
        out16 = self.decoder16(input)
        out32 = self.decoder32(out16)
        out64 = self.decoder64(out32)
        out128 = self.decoder128(out64)
        out256 = self.decoder256(out128)
        return out16, out32, out64, out128, out256


class appearDec128(nn.Module):

    def __init__(self, input_c, norm_layer, size_=256):
        super(appearDec128, self).__init__()
        layers = []
        channel_list = [1024, 1024, 1024]
        c0 = 1024
        for cc in channel_list:
            layers.append(nn.ConvTranspose2d(c0, cc, 4, 2, 1))
            layers.append(norm_layer(cc))
            layers.append(nn.ReLU(True))
            c0 = cc
        self.decoder8 = nn.Sequential(*layers)
        self.decoder16 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 2, 1), norm_layer(512), nn.ReLU(True))
        self.decoder32 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), norm_layer(256), nn.ReLU(True))
        self.decoder64 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), norm_layer(128), nn.ReLU(True))
        layers = []
        layers.append(nn.ConvTranspose2d(128, 3, 4, 2, 1))
        layers.append(nn.Tanh())
        self.decoder128 = nn.Sequential(*layers)

    def forward(self, input):
        out8 = self.decoder8(input)
        out16 = self.decoder16(out8)
        out32 = self.decoder32(out16)
        out64 = self.decoder64(out32)
        out128 = self.decoder128(out64)
        return out8, out16, out32, out64, out128


class appearEnc(nn.Module):

    def __init__(self, input_c, norm_layer, size_=256, conv_k=4):
        super(appearEnc, self).__init__()
        layers = []
        channel_list = [128, 256, 512, 1024, 1024, 1024]
        c0 = 64
        layers.append(nn.Conv2d(input_c, c0, conv_k, 2, 1))
        layers.append(nn.LeakyReLU(0.2))
        for cc in channel_list:
            layers.append(nn.Conv2d(c0, cc, conv_k, 2, 1))
            layers.append(norm_layer(cc))
            layers.append(nn.LeakyReLU(0.2))
            c0 = cc
        self.encoder = nn.Sequential(*layers)
        layers = []
        layers.append(nn.Conv2d(1024, 1024, conv_k, 2, 1))
        self.mean = nn.Sequential(*layers)

    def sample_z(self, z_mu):
        z_std = 1.0
        eps = th.randn(z_mu.size()).type_as(z_mu)
        return z_mu + z_std * eps

    def kl_loss(self, z_mu):
        z_var = th.ones(z_mu.size()).type_as(z_mu)
        kl_loss_ = th.mean(0.5 * th.sum(th.exp(z_var) + z_mu ** 2 - 1.0 - z_var, 1))
        return kl_loss_

    def freeze(self):
        for module_ in self.encoder:
            for p in module_.parameters():
                p.requires_grad = False
        for module_ in self.mean:
            for p in module_.parameters():
                p.requires_grad = False

    def forward(self, input):
        encoder = self.encoder(input)
        z_mu = self.mean(encoder)
        sample_z = self.sample_z(z_mu)
        kl_loss = self.kl_loss(z_mu)
        return sample_z, kl_loss, z_mu


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.net = [nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias), norm_layer(ndf * 2), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        if use_sigmoid:
            self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class MLP(nn.Module):

    def __init__(self, input_nc, output_nc):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_nc, output_nc)

    def forward(self, input):
        return self.fc(input)


class ArcMarginProduct(nn.Module):
    """Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.s
        return output


class AddMarginProduct(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.s
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.in_features) + ', out_features=' + str(self.out_features) + ', s=' + str(self.s) + ', m=' + str(self.m) + ')'


class SphereProduct(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """

    def __init__(self, in_features, out_features, m=4):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)
        self.mlambda = [lambda x: x ** 0, lambda x: x ** 1, lambda x: 2 * x ** 2 - 1, lambda x: 4 * x ** 3 - 3 * x, lambda x: 8 * x ** 4 - 8 * x ** 2 + 1, lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x]

    def forward(self, input, label):
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = (-1.0) ** k * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = one_hot * (phi_theta - cos_theta) / (1 + self.lamb) + cos_theta
        output *= NormOfFeature.view(-1, 1)
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.in_features) + ', out_features=' + str(self.out_features) + ', m=' + str(self.m) + ')'


model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


class faceIDNet(nn.Module):

    def __init__(self, input_nc, class_num):
        super(faceIDNet, self).__init__()
        self.feat = resnet18(input_nc, use_se=False)
        self.mlp = MLP(512, class_num)

    def forward(self, input):
        feat = self.feat(input)
        pred = self.mlp(feat)
        return pred

    def face_id_loss(self, x, target, loss_func):
        targetIdFeat256 = self.feat(target).detach()
        faceIDFeat = self.feat(x)
        id_loss = loss_func(faceIDFeat, targetIdFeat256)
        return id_loss


class BasicSPADE(nn.Module):

    def __init__(self, norm_layer, input_nc, planes):
        super(BasicSPADE, self).__init__()
        self.conv_weight = nn.Conv2d(input_nc, planes, kernel_size=3, stride=1, padding=1)
        self.conv_bias = nn.Conv2d(input_nc, planes, kernel_size=3, stride=1, padding=1)
        self.norm = norm_layer(planes, affine=False)

    def forward(self, x, bound):
        out = self.norm(x)
        weight_norm = self.conv_weight(bound)
        bias_norm = self.conv_bias(bound)
        out = out * weight_norm + bias_norm
        return out


class ResBlkSPADE(nn.Module):

    def __init__(self, norm_layer, input_nc, planes, conv_kernel_size=1, padding=0):
        super(ResBlkSPADE, self).__init__()
        self.spade1 = BasicSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.spade2 = BasicSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.spade_res = BasicSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=planes)
        self.conv_res = nn.Conv2d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding)

    def forward(self, x, bound):
        out = self.spade1(x, bound)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.spade2(out, bound)
        out = self.relu(out)
        out = self.conv2(out)
        residual = x
        residual = self.spade_res(residual, bound)
        residual = self.relu(residual)
        residual = self.conv_res(residual)
        out = out + residual
        return out


class SPADEGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, latent_chl=1024, up_mode='NF'):
        super(SPADEGenerator, self).__init__()
        layers = []
        self.up_mode = up_mode
        self.up1 = nn.ConvTranspose2d(in_channels=latent_chl, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.up5 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.up6 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.up7 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.up8 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.spade_blc3 = ResBlkSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=512)
        self.spade_blc4 = ResBlkSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=512)
        self.spade_blc5 = ResBlkSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=512)
        self.spade_blc6 = ResBlkSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=256)
        self.spade_blc7 = ResBlkSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=128)
        self.spade_blc8 = ResBlkSPADE(norm_layer=norm_layer, input_nc=input_nc, planes=64)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.same = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, input, latent_z):
        bound128 = F.interpolate(input, scale_factor=0.5)
        bound64 = F.interpolate(bound128, scale_factor=0.5)
        bound32 = F.interpolate(bound64, scale_factor=0.5)
        bound16 = F.interpolate(bound32, scale_factor=0.5)
        bound8 = F.interpolate(bound16, scale_factor=0.5)
        bound4 = F.interpolate(bound8, scale_factor=0.5)
        x_up1 = self.up1(latent_z)
        x_up2 = self.up2(x_up1)
        x_up3 = self.spade_blc3(x_up2, bound4)
        x_up3 = self.up3(x_up3)
        x_up4 = self.spade_blc4(x_up3, bound8)
        x_up4 = self.up4(x_up4)
        x_up5 = self.spade_blc5(x_up4, bound16)
        x_up5 = self.conv5(x_up5)
        x_up5 = self.up5(x_up5)
        x_up6 = self.spade_blc6(x_up5, bound32)
        x_up6 = self.conv6(x_up6)
        x_up6 = self.up6(x_up6)
        x_up7 = self.spade_blc7(x_up6, bound64)
        x_up7 = self.conv7(x_up7)
        x_up7 = self.up7(x_up7)
        x_up8 = self.spade_blc8(x_up7, bound128)
        x_up8 = self.conv8(x_up8)
        x_up8 = self.up8(x_up8)
        x_out = self.same(x_up8)
        x_out = self.tanh(x_out)
        return x_out


def gram_matrix(feat):
    b, ch, h, w = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = th.bmm(feat_t, feat) / (ch * h * w)
    gram = gram.view(b, h, w)
    return gram


class VGGNet(nn.Module):

    def __init__(self):
        super(VGGNet, self).__init__()
        self.net = vgg16()
        vgg_path = 'pretrainModel/vgg16-397923af.pth'
        self.net.load_state_dict(th.load(vgg_path))

    def forward(self, x):
        map_ = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        vgg_layers = self.net.features
        layer_name_mapping = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}
        output = []
        for name, module in vgg_layers._modules.items():
            x = module(x)
            if name in layer_name_mapping:
                output.append(x)
        return output

    def perceptual_loss(self, x, target, loss_func):
        self.x_result = self.forward(x)
        self.target_result = self.forward(target)
        loss_ = 0
        for xx, yy in zip(self.x_result, self.target_result):
            loss_ += loss_func(xx, yy.detach())
        return loss_

    def style_loss(self, x, target, loss_func):
        loss_ = 0
        for xx, yy in zip(self.x_result, self.target_result):
            loss_ += loss_func(gram_matrix(xx), gram_matrix(yy.detach()))
        return loss_


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicSPADE,
     lambda: ([], {'norm_layer': _mock_layer, 'input_nc': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GANLoss,
     lambda: ([], {}),
     lambda: ([], {'input': torch.rand([4, 4]), 'target_is_real': 4}),
     True),
    (MLP,
     lambda: ([], {'input_nc': 4, 'output_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NLayerDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (PixelDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlkSPADE,
     lambda: ([], {'norm_layer': _mock_layer, 'input_nc': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SEBlock,
     lambda: ([], {'channel': 16}),
     lambda: ([torch.rand([4, 16, 4, 16])], {}),
     True),
    (appearDec,
     lambda: ([], {'input_c': 4, 'norm_layer': _mock_layer}),
     lambda: ([torch.rand([4, 1024, 4, 4])], {}),
     True),
    (appearDec128,
     lambda: ([], {'input_c': 4, 'norm_layer': _mock_layer}),
     lambda: ([torch.rand([4, 1024, 4, 4])], {}),
     True),
    (appearEnc,
     lambda: ([], {'input_c': 4, 'norm_layer': _mock_layer}),
     lambda: ([torch.rand([4, 4, 256, 256])], {}),
     True),
]

class Test_bj80heyue_One_Shot_Face_Reenactment(_paritybench_base):
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

