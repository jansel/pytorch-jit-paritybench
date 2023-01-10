import sys
_module = sys.modules[__name__]
del sys
pred_benchmark = _module
train_benchmark = _module
conf = _module
elektronn3 = _module
_version = _module
data = _module
cnndata = _module
coord_transforms = _module
knossos = _module
knossos_labels = _module
sources = _module
transforms = _module
functional = _module
random = _module
random_blurring = _module
region_generator = _module
transforms = _module
utils = _module
inference = _module
inference = _module
logger = _module
models = _module
_model_utils = _module
base = _module
fcn = _module
fcn_2d = _module
msdnet = _module
resunet = _module
simple = _module
tiramisu_2d = _module
unet = _module
unet3d_lite = _module
vnet = _module
modules = _module
axial_attention = _module
evonorm = _module
l1batchnorm = _module
layers = _module
loss = _module
lovasz_losses = _module
wsconv = _module
training = _module
_trainer_multi = _module
handlers = _module
metrics = _module
noise2void = _module
padam = _module
plotting = _module
recalibration = _module
swa = _module
train_utils = _module
trainer = _module
trainer_gnn = _module
trainer_gnn_batch = _module
trainer_gnn_minibatch = _module
triplettrainer = _module
inference_h5 = _module
train_noise2void = _module
train_simple2d = _module
train_unet_neurodata = _module
validate = _module
setup = _module
versioneer = _module

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


import time


import torch


import numpy as np


import logging


import random


from torch import nn


from torch import optim


from typing import Tuple


from typing import Dict


from typing import Optional


from typing import Union


from typing import Sequence


from typing import Any


from typing import List


from typing import Callable


from torch.utils import data


import torch.utils.data


import collections


import warnings


from scipy.ndimage.filters import gaussian_filter


from scipy.ndimage.interpolation import map_coordinates


from scipy.ndimage.morphology import distance_transform_edt


import copy


import itertools


from collections import OrderedDict


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torchvision import models


from torchvision.models.vgg import VGG


from torch.utils.checkpoint import checkpoint


from torch.nn import functional as F


from torch.autograd.function import Function


from torch.utils.checkpoint import get_device_states


from torch.utils.checkpoint import set_device_states


from torch.autograd import Variable


from torch.functional import F


from torch import Tensor


import inspect


from math import nan


import matplotlib.figure


import matplotlib.pyplot as plt


import matplotlib.cm


from functools import lru_cache


import sklearn.metrics


from torch.cuda import amp


from torch.optim import Optimizer


import math


from collections import defaultdict


from itertools import chain


from collections import deque


from itertools import islice


from torch.utils.data import DataLoader


from torch.optim.lr_scheduler import StepLR


from torch.utils import collect_env


import matplotlib


import torch.nn


from torch.utils.tensorboard import SummaryWriter


from sklearn.manifold import TSNE


from sklearn.cluster import KMeans


from sklearn.metrics import v_measure_score


from sklearn.cluster import MiniBatchKMeans


from sklearn.linear_model import SGDClassifier


from sklearn.metrics.cluster import homogeneity_completeness_v_measure


class Argmax(nn.Module):

    def __init__(self, dim=1, unsqueeze=True):
        super().__init__()
        self.dim = dim
        self.unsqueeze = unsqueeze

    def forward(self, x):
        argmax = torch.argmax(x, self.dim)
        if self.unsqueeze:
            argmax.unsqueeze_(1)
        return argmax


class fcn32s(nn.Module):

    def __init__(self, n_classes=2, learned_billinear=False, red_fac=16):
        super(fcn32s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.conv_block1 = nn.Sequential(nn.Conv3d(1, 64 // red_fac, 3, padding=100), nn.ReLU(inplace=True), nn.MaxPool3d(2, stride=2, ceil_mode=True))
        self.conv_block2 = nn.Sequential(nn.Conv3d(64 // red_fac, 128 // red_fac, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool3d(2, stride=2, ceil_mode=True))
        self.conv_block3 = nn.Sequential(nn.Conv3d(128 // red_fac, 256 // red_fac, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool3d(2, stride=2, ceil_mode=True))
        self.conv_block4 = nn.Sequential(nn.Conv3d(256 // red_fac, 512 // red_fac, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool3d(2, stride=2, ceil_mode=True))
        self.conv_block5 = nn.Sequential(nn.Conv3d(512 // red_fac, 512 // red_fac, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool3d(2, stride=2, ceil_mode=True))
        self.classifier = nn.Sequential(nn.Conv3d(512 // red_fac, 4096 // red_fac, 7), nn.ReLU(inplace=True), nn.Dropout3d(), nn.Conv3d(4096 // red_fac, self.n_classes, 1))
        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        out = self.conv_block4(out)
        out = self.conv_block5(out)
        score = self.classifier(out)
        out = F.upsample(score, x.size()[2:], mode='trilinear')
        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5]
        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv3d) and isinstance(l2, nn.Conv3d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]


class fcn16s(nn.Module):

    def __init__(self, n_classes=2, learned_billinear=False):
        super(fcn16s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.conv_block1 = nn.Sequential(nn.Conv3d(1, 64, 3, padding=100), nn.ReLU(inplace=True), nn.MaxPool3d(2, stride=2, ceil_mode=True))
        self.conv_block2 = nn.Sequential(nn.Conv3d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool3d(2, stride=2, ceil_mode=True))
        self.conv_block3 = nn.Sequential(nn.Conv3d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool3d(2, stride=2, ceil_mode=True))
        self.conv_block4 = nn.Sequential(nn.Conv3d(256, 512, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool3d(2, stride=2, ceil_mode=True))
        self.conv_block5 = nn.Sequential(nn.Conv3d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool3d(2, stride=2, ceil_mode=True))
        self.classifier = nn.Sequential(nn.Conv3d(512, 4096, 7), nn.ReLU(inplace=True), nn.Dropout3d(), nn.Conv3d(4096, self.n_classes, 1))
        self.score_pool4 = nn.Conv3d(512, self.n_classes, 1)
        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)
        score = F.upsample(score, score_pool4.size()[2:], mode='trilinear')
        score += score_pool4
        out = F.upsample(score, x.size()[2:], mode='trilinear')
        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5]
        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv3d) and isinstance(l2, nn.Conv3d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]


class fcn8s(nn.Module):

    def __init__(self, n_classes=2, learned_billinear=False):
        super(fcn8s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.conv_block1 = nn.Sequential(nn.Conv3d(1, 64, 3, padding=100), nn.ReLU(inplace=True), nn.MaxPool3d(2, stride=2, ceil_mode=True))
        self.conv_block2 = nn.Sequential(nn.Conv3d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool3d(2, stride=2, ceil_mode=True))
        self.conv_block3 = nn.Sequential(nn.Conv3d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool3d(2, stride=2, ceil_mode=True))
        self.conv_block4 = nn.Sequential(nn.Conv3d(256, 512, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool3d(2, stride=2, ceil_mode=True))
        self.conv_block5 = nn.Sequential(nn.Conv3d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool3d(2, stride=2, ceil_mode=True))
        self.classifier = nn.Sequential(nn.Conv3d(512, 4096, 7), nn.ReLU(inplace=True), nn.Dropout3d(), nn.Conv3d(4096, self.n_classes, 1))
        self.score_pool4 = nn.Conv3d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv3d(256, self.n_classes, 1)
        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)
        score_pool3 = self.score_pool3(conv3)
        score = F.upsample(score, score_pool4.size()[2:], mode='trilinear')
        score += score_pool4
        score = F.upsample(score, score_pool3.size()[2:], mode='trilinear')
        score += score_pool3
        out = F.upsample(score, x.size()[2:], mode='trilinear')
        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5]
        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv3d) and isinstance(l2, nn.Conv3d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]


class FCN32s(nn.Module):

    def __init__(self, base_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.base_net = base_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.base_net(x)
        x5 = output['x5']
        score = self.bn1(self.relu(self.deconv1(x5)))
        score = self.bn2(self.relu(self.deconv2(score)))
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)
        return score


class FCN16s(nn.Module):

    def __init__(self, base_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.base_net = base_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.base_net(x)
        x5 = output['x5']
        x4 = output['x4']
        score = self.relu(self.deconv1(x5))
        score = self.bn1(score + x4)
        score = self.bn2(self.relu(self.deconv2(score)))
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)
        return score


class FCN8s(nn.Module):

    def __init__(self, base_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.base_net = base_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.base_net(x)
        x5 = output['x5']
        x4 = output['x4']
        x3 = output['x3']
        score = self.relu(self.deconv1(x5))
        score = self.bn1(score + x4)
        score = self.relu(self.deconv2(score))
        score = self.bn2(score + x3)
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)
        return score


class FCNs(nn.Module):

    def __init__(self, base_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.base_net = base_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.base_net(x)
        x5 = output['x5']
        x4 = output['x4']
        x3 = output['x3']
        x2 = output['x2']
        x1 = output['x1']
        score = self.bn1(self.relu(self.deconv1(x5)))
        score = score + x4
        score = self.bn2(self.relu(self.deconv2(score)))
        score = score + x3
        score = self.bn3(self.relu(self.deconv3(score)))
        score = score + x2
        score = self.bn4(self.relu(self.deconv4(score)))
        score = score + x1
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)
        return score


def add_conv_block(in_ch=1, out_ch=1, kernel_size=3, dilate=1, last=False, volumetric=True):
    if volumetric:
        Conv = nn.Conv3d
        BatchNorm = nn.BatchNorm3d
    else:
        Conv = nn.Conv2d
        BatchNorm = nn.BatchNorm2d
    pad = dilate if not last else 0
    conv_1 = Conv(in_ch, out_ch, kernel_size, padding=pad, dilation=dilate)
    bn_1 = BatchNorm(out_ch)
    return [conv_1, bn_1]


class MSDNet(nn.Module):
    """
    Paper: A mixed-scale dense convolutional neural network for image analysis
    Published: PNAS, Jan. 2018 
    Paper: http://www.pnas.org/content/early/2017/12/21/1715832114
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal(m, m.weight.data)

    def __init__(self, in_channels=1, out_channels=2, num_layers=40, volumetric=True):
        super().__init__()
        self.layer_list = add_conv_block(in_ch=in_channels, volumetric=volumetric)
        current_in_channels = 1
        for i in range(num_layers):
            s = i % 10 + 1
            self.layer_list += add_conv_block(in_ch=current_in_channels, dilate=s, volumetric=volumetric)
            current_in_channels += 1
        self.layer_list += add_conv_block(in_ch=current_in_channels + in_channels, out_ch=out_channels, kernel_size=1, last=True, volumetric=volumetric)
        self.layers = nn.ModuleList(self.layer_list)
        self.apply(self.weight_init)

    def forward(self, x):
        prev_features = []
        inp = x
        for i, f in enumerate(self.layers):
            if i == len(self.layers) - 2:
                x = torch.cat(prev_features + [inp], 1)
            x = f(x)
            if (i + 1) % 2 == 0 and not i == len(self.layers) - 1:
                x = F.relu(x)
                prev_features.append(x)
                x = torch.cat(prev_features, 1)
        return x


def get_activation(activation):
    if isinstance(activation, str):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky':
            return nn.LeakyReLU(negative_slope=0.1)
        elif activation == 'prelu':
            return nn.PReLU(num_parameters=1)
        elif activation == 'rrelu':
            return nn.RReLU()
        elif activation == 'silu':
            return nn.SiLU()
        elif activation == 'lin':
            return nn.Identity()
    else:
        return copy.deepcopy(activation)


def get_conv(dim=3):
    """Chooses an implementation for a convolution layer."""
    if dim == 3:
        return nn.Conv3d
    elif dim == 2:
        return nn.Conv2d
    else:
        raise ValueError('dim has to be 2 or 3')


def get_normalization(normtype: str, num_channels: int, dim: int=3):
    """Chooses an implementation for a batch normalization layer."""
    if normtype is None or normtype == 'none':
        return nn.Identity()
    elif normtype.startswith('group'):
        if normtype == 'group':
            num_groups = 8
        elif len(normtype) > len('group') and normtype[len('group'):].isdigit():
            num_groups = int(normtype[len('group'):])
        else:
            raise ValueError(f'normtype "{normtype}" not understood. It should be "group<G>", where <G> is the number of groups.')
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif normtype == 'instance':
        if dim == 3:
            return nn.InstanceNorm3d(num_channels)
        elif dim == 2:
            return nn.InstanceNorm2d(num_channels)
        else:
            raise ValueError('dim has to be 2 or 3')
    elif normtype == 'batch':
        if dim == 3:
            return nn.BatchNorm3d(num_channels)
        elif dim == 2:
            return nn.BatchNorm2d(num_channels)
        else:
            raise ValueError('dim has to be 2 or 3')
    else:
        raise ValueError(f'Unknown normalization type "{normtype}".\nValid choices are "batch", "instance", "group" or "group<G>",where <G> is the number of groups.')


def get_padding(conv_mode, kernel_size):
    if conv_mode == 'valid' or kernel_size == 1:
        return 0
    elif conv_mode == 'same' and kernel_size == 3:
        return 1
    else:
        raise NotImplementedError(f'conv_mode {conv_mode} with kernel_size {kernel_size} unsupported.')


def planar_kernel(x):
    """Returns a "planar" kernel shape (e.g. for 2D convolution in 3D space)
    that doesn't consider the first spatial dim (D)."""
    if isinstance(x, int):
        return 1, x, x
    else:
        return x


def planar_pad(x):
    """Returns a "planar" padding shape that doesn't pad along the first spatial dim (D)."""
    if isinstance(x, int):
        return 0, x, x
    else:
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, planar=False, activation='relu', normalization=None, dim=3, conv_mode='same', residual=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.conv_mode = conv_mode
        self.activation = activation
        self.residual = residual
        self.dim = dim
        padding = get_padding(conv_mode, kernel_size)
        if planar:
            padding = planar_pad(padding)
            kernel_size = planar_kernel(kernel_size)
        conv_class = get_conv(dim)
        self.conv1 = conv_class(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = get_normalization(normalization, self.out_channels, dim=dim)
        self.act1 = get_activation(activation)
        self.conv2 = conv_class(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = get_normalization(normalization, self.out_channels, dim=dim)
        self.act2 = get_activation(activation)
        if self.residual and self.in_channels != self.out_channels:
            self.proj = conv_class(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, inp):
        y = self.conv1(inp)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.conv2(y)
        if self.residual:
            y += self.proj(inp)
        y = self.norm2(y)
        y = self.act2(y)
        return y


def get_maxpool(dim=3):
    """Chooses an implementation for a max-pooling layer."""
    if dim == 3:
        return nn.MaxPool3d
    elif dim == 2:
        return nn.MaxPool2d
    else:
        raise ValueError('dim has to be 2 or 3')


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pooling=True, planar=False, activation='relu', normalization=None, dim=3, conv_mode='same', res_blocks=0, skip_first_residual=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.normalization = normalization
        self.res_blocks = res_blocks
        self.dim = dim
        enable_residual = res_blocks >= 1
        convs = [ConvBlock(self.in_channels, self.out_channels, planar=planar, activation=activation, normalization=normalization, conv_mode=conv_mode, residual=enable_residual and not skip_first_residual)]
        for _ in range(res_blocks - 1):
            convs.append(ConvBlock(self.out_channels, self.out_channels, planar=planar, activation=activation, normalization=normalization, conv_mode=conv_mode, residual=enable_residual))
        self.convs = nn.Sequential(*convs)
        if pooling:
            pool_ks = planar_kernel(2) if planar else 2
            self.pool = get_maxpool(dim)(kernel_size=pool_ks, ceil_mode=True)
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        y = self.convs(x)
        before_pool = y
        y = self.pool(y)
        return y, before_pool


class DummyAttention(nn.Module):

    def forward(self, x, g):
        return x, None


class GridAttention(nn.Module):
    """Based on https://github.com/ozan-oktay/Attention-Gated-Networks

    Published in https://arxiv.org/abs/1804.03999"""

    def __init__(self, in_channels, gating_channels, inter_channels=None, dim=3, sub_sample_factor=2):
        super().__init__()
        assert dim in [2, 3]
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dim
        self.dim = dim
        self.sub_sample_kernel_size = self.sub_sample_factor
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        if dim == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dim == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplementedError
        self.w = nn.Sequential(conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1), bn(self.in_channels))
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, bias=True)
        self.init_weights()

    def forward(self, x, g):
        theta_x = self.theta(x)
        phi_g = F.interpolate(self.phi(g), size=theta_x.shape[2:], mode=self.upsample_mode, align_corners=False)
        f = F.relu(theta_x + phi_g, inplace=True)
        sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(sigm_psi_f, size=x.shape[2:], mode=self.upsample_mode, align_corners=False)
        y = sigm_psi_f.expand_as(x) * x
        wy = self.w(y)
        return wy, sigm_psi_f

    def init_weights(self):

        def weight_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif classname.find('Linear') != -1:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
        self.apply(weight_init)


@torch.jit.script
def autocrop(from_down: torch.Tensor, from_up: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Crops feature tensors from the encoder and decoder pathways so that they
    can be combined.

    - If inputs from the encoder pathway have shapes that are not divisible
      by 2, the use of ``nn.MaxPool(ceil_mode=True)`` leads to the 2x
      upconvolution results being too large by one element in each odd
      dimension, so they need to be cropped in these dimensions.

    - If VALID convolutions are used, feature tensors get smaller with each
      convolution, so we need to center-crop the larger feature tensors from
      the encoder pathway to make features combinable with the smaller
      decoder feautures.

    Args:
        from_down: Feature from encoder pathway (``DownConv``)
        from_up: Feature from decoder pathway (2x upsampled)

    Returns:

    """
    ndim = from_down.dim()
    if from_down.shape[2:] == from_up.shape[2:]:
        return from_down, from_up
    ds = from_down.shape[2:]
    us = from_up.shape[2:]
    upcrop = [(u - (u - d) % 2) for d, u in zip(ds, us)]
    if ndim == 4:
        from_up = from_up[:, :, :upcrop[0], :upcrop[1]]
    if ndim == 5:
        from_up = from_up[:, :, :upcrop[0], :upcrop[1], :upcrop[2]]
    ds = from_down.shape[2:]
    us = from_up.shape[2:]
    assert ds[0] >= us[0], f'{ds, us}'
    assert ds[1] >= us[1]
    if ndim == 4:
        from_down = from_down[:, :, (ds[0] - us[0]) // 2:(ds[0] + us[0]) // 2, (ds[1] - us[1]) // 2:(ds[1] + us[1]) // 2]
    elif ndim == 5:
        assert ds[2] >= us[2]
        from_down = from_down[:, :, (ds[0] - us[0]) // 2:(ds[0] + us[0]) // 2, (ds[1] - us[1]) // 2:(ds[1] + us[1]) // 2, (ds[2] - us[2]) // 2:(ds[2] + us[2]) // 2]
    return from_down, from_up


def conv1(in_channels, out_channels, dim=3):
    """Returns a 1x1 or 1x1x1 convolution, depending on dim"""
    return get_conv(dim)(in_channels, out_channels, kernel_size=1)


def conv3(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, planar=False, dim=3):
    """Returns an appropriate spatial convolution layer, depending on args.
    - dim=2: Conv2d with 3x3 kernel
    - dim=3 and planar=False: Conv3d with 3x3x3 kernel
    - dim=3 and planar=True: Conv3d with 1x3x3 kernel
    """
    if planar:
        stride = planar_kernel(stride)
        padding = planar_pad(padding)
        kernel_size = planar_kernel(kernel_size)
    return get_conv(dim)(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


class ResizeConv(nn.Module):
    """Upsamples by 2x and applies a convolution.

    This is meant as a replacement for transposed convolution to avoid
    checkerboard artifacts. See

    - https://distill.pub/2016/deconv-checkerboard/
    - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, planar=False, dim=3, upsampling_mode='nearest'):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.scale_factor = 2
        if dim == 3 and planar:
            self.scale_factor = planar_kernel(self.scale_factor)
        self.dim = dim
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode=self.upsampling_mode)
        if kernel_size == 3:
            self.conv = conv3(in_channels, out_channels, padding=1, planar=planar, dim=dim)
        elif kernel_size == 1:
            self.conv = conv1(in_channels, out_channels, dim=dim)
        else:
            raise ValueError(f'kernel_size={kernel_size} is not supported. Choose 1 or 3.')

    def forward(self, x):
        return self.conv(self.upsample(x))


def get_convtranspose(dim=3):
    """Chooses an implementation for a transposed convolution layer."""
    if dim == 3:
        return nn.ConvTranspose3d
    elif dim == 2:
        return nn.ConvTranspose2d
    else:
        raise ValueError('dim has to be 2 or 3')


def upconv2(in_channels, out_channels, mode='transpose', planar=False, dim=3):
    """Returns a learned upsampling operator depending on args."""
    kernel_size = 2
    stride = 2
    if planar:
        kernel_size = planar_kernel(kernel_size)
        stride = planar_kernel(stride)
    if mode == 'transpose':
        return get_convtranspose(dim)(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    elif 'resizeconv' in mode:
        if 'linear' in mode:
            upsampling_mode = 'trilinear' if dim == 3 else 'bilinear'
        else:
            upsampling_mode = 'nearest'
        rc_kernel_size = 1 if mode.endswith('1') else 3
        return ResizeConv(in_channels, out_channels, planar=planar, dim=dim, upsampling_mode=upsampling_mode, kernel_size=rc_kernel_size)


class UpBlock(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    att: Optional[torch.Tensor]

    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose', planar=False, activation='relu', normalization=None, full_norm=True, dim=3, conv_mode='same', attention=False, res_blocks=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.normalization = normalization
        self.res_blocks = res_blocks
        self.dim = dim
        enable_residual = res_blocks >= 1
        self.upconv = upconv2(self.in_channels, self.out_channels, mode=self.up_mode, planar=planar, dim=dim)
        self.act0 = get_activation(activation)
        self.norm0 = get_normalization(normalization, out_channels, dim=dim)
        if attention:
            self.attention = GridAttention(in_channels=in_channels // 2, gating_channels=in_channels, dim=dim)
        else:
            self.attention = DummyAttention()
        self.att = None
        if self.merge_mode == 'concat':
            convs = [ConvBlock(2 * self.out_channels, self.out_channels, planar=planar, activation=activation, normalization=normalization, conv_mode=conv_mode, residual=enable_residual)]
        else:
            convs = [ConvBlock(self.out_channels, self.out_channels, planar=planar, activation=activation, normalization=normalization, conv_mode=conv_mode, residual=enable_residual)]
        for _ in range(res_blocks - 1):
            convs.append(ConvBlock(self.out_channels, self.out_channels, planar=planar, activation=activation, normalization=normalization, conv_mode=conv_mode, residual=enable_residual))
        self.convs = nn.Sequential(*convs)

    def forward(self, enc, dec):
        """ Forward pass
        Arguments:
            enc: Tensor from the encoder pathway
            dec: Tensor from the decoder pathway (to be upconv'd)
        """
        updec = self.upconv(dec)
        enc, updec = autocrop(enc, updec)
        genc, att = self.attention(enc, dec)
        if not torch.jit.is_scripting():
            self.att = att
        updec = self.norm0(updec)
        updec = self.act0(updec)
        if self.merge_mode == 'concat':
            mrg = torch.cat((updec, genc), 1)
        else:
            mrg = updec + genc
        y = self.convs(mrg)
        return y


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True, planar=False, activation='relu', normalization=None, full_norm=True, dim=3, conv_mode='same'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.normalization = normalization
        self.dim = dim
        padding = 1 if 'same' in conv_mode else 0
        self.conv1 = conv3(self.in_channels, self.out_channels, planar=planar, dim=dim, padding=padding)
        self.conv2 = conv3(self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding)
        if self.pooling:
            kernel_size = 2
            if planar:
                kernel_size = planar_kernel(kernel_size)
            self.pool = get_maxpool(dim)(kernel_size=kernel_size, ceil_mode=True)
            self.pool_ks = kernel_size
        else:
            self.pool = nn.Identity()
            self.pool_ks = -123
        self.act1 = get_activation(activation)
        self.act2 = get_activation(activation)
        if full_norm:
            self.norm0 = get_normalization(normalization, self.out_channels, dim=dim)
        else:
            self.norm0 = nn.Identity()
        self.norm1 = get_normalization(normalization, self.out_channels, dim=dim)

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm0(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.norm1(y)
        y = self.act2(y)
        before_pool = y
        y = self.pool(y)
        return y, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    att: Optional[torch.Tensor]

    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose', planar=False, activation='relu', normalization=None, full_norm=True, dim=3, conv_mode='same', attention=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.normalization = normalization
        padding = 1 if 'same' in conv_mode else 0
        self.upconv = upconv2(self.in_channels, self.out_channels, mode=self.up_mode, planar=planar, dim=dim)
        if self.merge_mode == 'concat':
            self.conv1 = conv3(2 * self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding)
        else:
            self.conv1 = conv3(self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding)
        self.conv2 = conv3(self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding)
        self.act0 = get_activation(activation)
        self.act1 = get_activation(activation)
        self.act2 = get_activation(activation)
        if full_norm:
            self.norm0 = get_normalization(normalization, self.out_channels, dim=dim)
            self.norm1 = get_normalization(normalization, self.out_channels, dim=dim)
        else:
            self.norm0 = nn.Identity()
            self.norm1 = nn.Identity()
        self.norm2 = get_normalization(normalization, self.out_channels, dim=dim)
        if attention:
            self.attention = GridAttention(in_channels=in_channels // 2, gating_channels=in_channels, dim=dim)
        else:
            self.attention = DummyAttention()
        self.att = None

    def forward(self, enc, dec):
        """ Forward pass
        Arguments:
            enc: Tensor from the encoder pathway
            dec: Tensor from the decoder pathway (to be upconv'd)
        """
        updec = self.upconv(dec)
        enc, updec = autocrop(enc, updec)
        genc, att = self.attention(enc, dec)
        if not torch.jit.is_scripting():
            self.att = att
        updec = self.norm0(updec)
        updec = self.act0(updec)
        if self.merge_mode == 'concat':
            mrg = torch.cat((updec, genc), 1)
        else:
            mrg = updec + genc
        y = self.conv1(mrg)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.act2(y)
        return y


class UNet(nn.Module):
    """Modified version of U-Net, adapted for 3D biomedical image segmentation

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding, expansive pathway)
    about an input tensor is merged with information representing the
    localization of details (from the encoding, compressive pathway).

    - Original paper: https://arxiv.org/abs/1505.04597
    - Base implementation: https://github.com/jaxony/unet-pytorch


    Modifications to the original paper (@jaxony):

    - Padding is used in size-3-convolutions to prevent loss
      of border pixels.
    - Merging outputs does not require cropping due to (1).
    - Residual connections can be used by specifying
      UNet(merge_mode='add').
    - If non-parametric upsampling is used in the decoder
      pathway (specified by upmode='upsample'), then an
      additional 1x1 convolution occurs after upsampling
      to reduce channel dimensionality by a factor of 2.
      This channel halving happens with the convolution in
      the tranpose convolution (specified by upmode='transpose').

    Additional modifications (@mdraw):

    - Operates on 3D image data (5D tensors) instead of 2D data
    - Uses 3D convolution, 3D pooling etc. by default
    - Each network block pair (the two corresponding submodules in the
      encoder and decoder pathways) can be configured to either work
      in 3D or 2D mode (3D/2D convolution, pooling etc.)
      with the `planar_blocks` parameter.
      This is helpful for dealing with data anisotropy (commonly the
      depth axis has lower resolution in SBEM data sets, so it is not
      as important for convolution/pooling) and can reduce the complexity of
      models (parameter counts, speed, memory usage etc.).
      Note: If planar blocks are used, the input patch size should be
      adapted by reducing depth and increasing height and width of inputs.
    - Configurable activation function.
    - Optional normalization

    Gradient checkpointing can be used to reduce memory consumption while
    training. To make use of gradient checkpointing, just run the
    ``forward_gradcp()`` instead of the regular ``forward`` method.
    This makes the backward pass a bit slower, but the memory savings can be
    huge (usually around 20% - 50%, depending on hyperparameters). Checkpoints
    are made after each network *block*.
    See https://pytorch.org/docs/master/checkpoint.html and
    https://arxiv.org/abs/1604.06174 for more details.
    Gradient checkpointing is not supported in TorchScript mode.

    Args:
        in_channels: Number of input channels
            (e.g. 1 for single-grayscale inputs, 3 for RGB images)
            Default: 1
        out_channels: Number of output channels (in classification/semantic
            segmentation, this is the number of different classes).
            Default: 2
        n_blocks: Number of downsampling/convolution blocks (max-pooling)
            in the encoder pathway. The decoder (upsampling/upconvolution)
            pathway will consist of `n_blocks - 1` blocks.
            Increasing `n_blocks` has two major effects:

            - The network will be deeper
              (n + 1 -> 4 additional convolution layers)
            - Since each block causes one additional downsampling, more
              contextual information will be available for the network,
              enhancing the effective visual receptive field.
              (n + 1 -> receptive field is approximately doubled in each
              dimension, except in planar blocks, in which it is only
              doubled in the H and W image dimensions)

            **Important note**: Always make sure that the spatial shape of
            your input is divisible by the number of blocks, because
            else, concatenating downsampled features will fail.
        start_filts: Number of filters for the first convolution layer.
            Note: The filter counts of the later layers depend on the
            choice of `merge_mode`.
        up_mode: Upsampling method in the decoder pathway.
            Choices:

            - 'transpose' (default): Use transposed convolution
              ("Upconvolution")
            - 'resizeconv_nearest': Use resize-convolution with nearest-
              neighbor interpolation, as proposed in
              https://distill.pub/2016/deconv-checkerboard/
            - 'resizeconv_linear: Same as above, but with (bi-/tri-)linear
              interpolation
            - 'resizeconv_nearest1': Like 'resizeconv_nearest', but using a
              light-weight 1x1 convolution layer instead of a spatial convolution
            - 'resizeconv_linear1': Like 'resizeconv_nearest', but using a
              light-weight 1x1-convolution layer instead of a spatial convolution
        merge_mode: How the features from the encoder pathway should
            be combined with the decoder features.
            Choices:

            - 'concat' (default): Concatenate feature maps along the
              `C` axis, doubling the number of filters each block.
            - 'add': Directly add feature maps (like in ResNets).
              The number of filters thus stays constant in each block.

            Note: According to https://arxiv.org/abs/1701.03056, feature
            concatenation ('concat') generally leads to better model
            accuracy than 'add' in typical medical image segmentation
            tasks.
        planar_blocks: Each number i in this sequence leads to the i-th
            block being a "planar" block. This means that all image
            operations performed in the i-th block in the encoder pathway
            and its corresponding decoder counterpart disregard the depth
            (`D`) axis and only operate in 2D (`H`, `W`).
            This is helpful for dealing with data anisotropy (commonly the
            depth axis has lower resolution in SBEM data sets, so it is
            not as important for convolution/pooling) and can reduce the
            complexity of models (parameter counts, speed, memory usage
            etc.).
            Note: If planar blocks are used, the input patch size should
            be adapted by reducing depth and increasing height and
            width of inputs.
        activation: Name of the non-linear activation function that should be
            applied after each network layer.
            Choices (see https://arxiv.org/abs/1505.00853 for details):

            - 'relu' (default)
            - 'silu': Sigmoid Linear Unit (SiLU, aka Swish)
            - 'leaky': Leaky ReLU (slope 0.1)
            - 'prelu': Parametrized ReLU. Best for training accuracy, but
              tends to increase overfitting.
            - 'rrelu': Can improve generalization at the cost of training
              accuracy.
            - Or you can pass an nn.Module instance directly, e.g.
              ``activation=torch.nn.ReLU()``
        normalization: Type of normalization that should be applied at the end
            of each block. Note that it is applied after the activated conv
            layers, not before the activation. This scheme differs from the
            original batch normalization paper and the BN scheme of 3D U-Net,
            but it delivers better results this way
            (see https://redd.it/67gonq).
            Choices:

            - 'group' for group normalization (G=8)
            - 'group<G>' for group normalization with <G> groups
              (e.g. 'group16') for G=16
            - 'instance' for instance normalization
            - 'batch' for batch normalization (default)
            - 'none' or ``None`` for no normalization
        attention: If ``True``, use grid attention in the decoding pathway,
            as proposed in https://arxiv.org/abs/1804.03999.
            Default: ``False``.
        full_norm: If ``True`` (default), perform normalization after each
            (transposed) convolution in the network (which is what almost
            all published neural network architectures do).
            If ``False``, only normalize after the last convolution
            layer of each block, in order to save resources. This was also
            the default behavior before this option was introduced.
        dim: Spatial dimensionality of the network. Choices:

            - 3 (default): 3D mode. Every block fully works in 3D unless
              it is excluded by the ``planar_blocks`` setting.
              The network expects and operates on 5D input tensors
              (N, C, D, H, W).
            - 2: Every block and every operation works in 2D, expecting
              4D input tensors (N, C, H, W).
        conv_mode: Padding mode of convolutions. Choices:

            - 'same' (default): Use SAME-convolutions in every layer:
              zero-padding inputs so that all convolutions preserve spatial
              shapes and don't produce an offset at the boundaries.
            - 'valid': Use VALID-convolutions in every layer: no padding is
              used, so every convolution layer reduces spatial shape by 2 in
              each dimension. Intermediate feature maps of the encoder pathway
              are automatically cropped to compatible shapes so they can be
              merged with decoder features.
              Advantages:

              - Less resource consumption than SAME because feature maps
                have reduced sizes especially in deeper layers.
              - No "fake" data (that is, the zeros from the SAME-padding)
                is fed into the network. The output regions that are influenced
                by zero-padding naturally have worse quality, so they should
                be removed in post-processing if possible (see
                ``overlap_shape`` in :py:mod:`elektronn3.inference`).
                Using VALID convolutions prevents the unnecessary computation
                of these regions that need to be cut away anyways for
                high-quality tiled inference.
              - Avoids the issues described in https://arxiv.org/abs/1811.11718.
              - Since the network will not receive zero-padded inputs, it is
                not required to learn a robustness against artificial zeros
                being in the border regions of inputs. This should reduce the
                complexity of the learning task and allow the network to
                specialize better on understanding the actual, unaltered
                inputs (effectively requiring less parameters to fit).

              Disadvantages:

              - Using this mode poses some additional constraints on input
                sizes and requires you to center-crop your targets,
                so it's harder to use in practice than the 'same' mode.
              - In some cases it might be preferable to get low-quality
                outputs at image borders as opposed to getting no outputs at
                the borders. Most notably this is the case if you do training
                and inference not on small patches, but on complete images in
                a single step.
    """

    def __init__(self, in_channels: int=1, out_channels: int=2, n_blocks: int=3, start_filts: int=32, up_mode: str='transpose', merge_mode: str='concat', planar_blocks: Sequence=(), batch_norm: str='unset', attention: bool=False, activation: Union[str, nn.Module]='relu', normalization: str='batch', full_norm: bool=True, dim: int=3, conv_mode: str='same'):
        super().__init__()
        if n_blocks < 1:
            raise ValueError('n_blocks must be > 1.')
        if dim not in {2, 3}:
            raise ValueError('dim has to be 2 or 3')
        if dim == 2 and planar_blocks != ():
            raise ValueError("If dim=2, you can't use planar_blocks since everything will be planar (2-dimensional) anyways.\nEither set dim=3 or set planar_blocks=().")
        if up_mode in ('transpose', 'upsample', 'resizeconv_nearest', 'resizeconv_linear', 'resizeconv_nearest1', 'resizeconv_linear1'):
            self.up_mode = up_mode
        else:
            raise ValueError('"{}" is not a valid mode for upsampling'.format(up_mode))
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError('"{}" is not a valid mode formerging up and down paths. Only "concat" and "add" are allowed.'.format(up_mode))
        if 'resizeconv' in self.up_mode and self.merge_mode == 'add':
            raise ValueError('up_mode "resizeconv" is incompatible with merge_mode "add" at the moment because it doesn\'t make sense to use nearest neighbour to reduce n_blocks channels (by half).')
        if len(planar_blocks) > n_blocks:
            raise ValueError("planar_blocks can't be longer than n_blocks.")
        if planar_blocks and (max(planar_blocks) >= n_blocks or min(planar_blocks) < 0):
            raise ValueError('planar_blocks has invalid value range. All values have to beblock indices, meaning integers between 0 and (n_blocks - 1).')
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.n_blocks = n_blocks
        self.normalization = normalization
        self.attention = attention
        self.conv_mode = conv_mode
        self.activation = activation
        self.dim = dim
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        if batch_norm != 'unset':
            raise RuntimeError('The `batch_norm` option has been replaced with the more general `normalization` option.\nIf you still want to use batch normalization, set `normalization=batch` instead.')
        self.planar_blocks = planar_blocks
        for i in range(n_blocks):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * 2 ** i
            pooling = True if i < n_blocks - 1 else False
            planar = i in self.planar_blocks
            down_conv = DownConv(ins, outs, pooling=pooling, planar=planar, activation=activation, normalization=normalization, full_norm=full_norm, dim=dim, conv_mode=conv_mode)
            self.down_convs.append(down_conv)
        for i in range(n_blocks - 1):
            ins = outs
            outs = ins // 2
            planar = n_blocks - 2 - i in self.planar_blocks
            up_conv = UpConv(ins, outs, up_mode=up_mode, merge_mode=merge_mode, planar=planar, activation=activation, normalization=normalization, attention=attention, full_norm=full_norm, dim=dim, conv_mode=conv_mode)
            self.up_convs.append(up_conv)
        self.conv_final = conv1(outs, self.out_channels, dim=dim)
        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        if isinstance(m, GridAttention):
            return
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            if getattr(m, 'bias') is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoder_outs = []
        i = 0
        for module in self.down_convs:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
            i += 1
        i = 0
        for module in self.up_convs:
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)
            i += 1
        x = self.conv_final(x)
        return x

    @torch.jit.unused
    def forward_gradcp(self, x):
        """``forward()`` implementation with gradient checkpointing enabled.
        Apart from checkpointing, this behaves the same as ``forward()``."""
        encoder_outs = []
        i = 0
        for module in self.down_convs:
            x, before_pool = checkpoint(module, x)
            encoder_outs.append(before_pool)
            i += 1
        i = 0
        for module in self.up_convs:
            before_pool = encoder_outs[-(i + 2)]
            x = checkpoint(module, before_pool, x)
            i += 1
        x = self.conv_final(x)
        return x


class Simple3DNet(nn.Module):

    def __init__(self, n_out_channels=2):
        super().__init__()
        self.n_out_channels = n_out_channels
        self.conv = nn.Sequential(nn.Conv3d(1, 10, 3, padding=1), nn.ReLU(), nn.Conv3d(10, 10, 3, padding=1), nn.ReLU(), nn.Conv3d(10, n_out_channels, 1))

    def forward(self, x):
        x = self.conv(x)
        return x


class Extended3DNet(nn.Module):

    def __init__(self, n_out_channels=2):
        super().__init__()
        self.n_out_channels = n_out_channels
        self.conv = nn.Sequential(nn.Conv3d(1, 64, 5, padding=2), nn.ReLU(), nn.Conv3d(64, 64, 5, padding=2), nn.ReLU(), nn.MaxPool3d(2), nn.Conv3d(64, 64, 3, padding=2), nn.ReLU(), nn.Conv3d(64, 64, 3, padding=1), nn.ReLU(), nn.Conv3d(64, 64, 3, padding=0), nn.ReLU(), nn.Conv3d(64, n_out_channels, 1))

    def forward(self, x):
        original_size = x.size()[2:]
        x = self.conv(x)
        x = F.upsample(x, original_size)
        return x


class N3DNet(nn.Module):

    def __init__(self, n_out_channels=2):
        super().__init__()
        self.n_out_channels = n_out_channels
        self.neuro3d_seq = nn.Sequential(nn.Conv3d(1, 20, (1, 5, 5), padding=(0, 2, 2)), nn.ReLU(), nn.Conv3d(20, 30, (1, 5, 5), padding=(0, 2, 2)), nn.ReLU(), nn.MaxPool3d((2, 2, 2)), nn.Conv3d(30, 40, (1, 5, 5), padding=(0, 2, 2)), nn.ReLU(), nn.Conv3d(40, 80, (3, 3, 3), padding=(1, 1, 1)), nn.ReLU(), nn.Conv3d(80, 100, (3, 3, 3), padding=(1, 1, 1)), nn.ReLU(), nn.Conv3d(100, 150, (1, 3, 3), padding=(0, 1, 1)), nn.ReLU(), nn.Conv3d(150, 50, (1, 1, 1)), nn.ReLU(), nn.Conv3d(50, n_out_channels, (1, 1, 1)))

    def forward(self, x):
        original_size = x.size()[2:]
        x = self.neuro3d_seq(x)
        x = F.upsample(x, original_size)
        return x


class Conv3DLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, batch_norm=True, pooling=None, dropout_rate=None, act=None):
        super().__init__()
        if act is None:
            act = nn.ReLU()
        seq = [nn.Conv3d(in_channels, out_channels, kernel_size)]
        if batch_norm:
            seq += [nn.BatchNorm3d(out_channels)]
        seq += [act]
        if pooling is not None:
            seq += [nn.MaxPool3d(pooling)]
        if dropout_rate is not None:
            seq += [nn.Dropout3d(dropout_rate)]
        self.conv3_layer = nn.Sequential(*seq)

    def forward(self, x):
        return self.conv3_layer(x)


class StackedConv2Scalar(nn.Module):

    def __init__(self, in_channels, n_classes, dropout_rate=0.05, act='relu'):
        super().__init__()
        if act == 'relu':
            act = nn.ReLU()
        elif act == 'leaky_relu':
            act = nn.LeakyReLU()
        self.seq = nn.Sequential(Conv3DLayer(in_channels, 20, (1, 5, 5), pooling=(1, 2, 2), dropout_rate=dropout_rate, act=act), Conv3DLayer(20, 30, (1, 5, 5), pooling=(1, 2, 2), dropout_rate=dropout_rate, act=act), Conv3DLayer(30, 40, (1, 4, 4), pooling=(1, 2, 2), dropout_rate=dropout_rate, act=act), Conv3DLayer(40, 50, (1, 4, 4), pooling=(1, 2, 2), dropout_rate=dropout_rate, act=act), Conv3DLayer(50, 60, (1, 2, 2), pooling=(1, 2, 2), dropout_rate=dropout_rate, act=act), Conv3DLayer(60, 70, (1, 1, 1), pooling=(1, 2, 2), dropout_rate=dropout_rate, act=act), Conv3DLayer(70, 70, (1, 1, 1), pooling=(1, 1, 1), dropout_rate=dropout_rate, act=act))
        self.adaptavgpool = nn.AdaptiveAvgPool1d(100)
        self.fc = nn.Sequential(nn.Linear(100, 50), act, nn.Linear(50, 30), act, nn.Linear(30, n_classes))

    def forward(self, x):
        x = self.seq(x)
        x = x.view(x.size()[0], 1, -1)
        x = self.adaptavgpool(x)
        x = self.fc(x.squeeze(1))
        return x


class StackedConv2ScalarWithLatentAdd(nn.Module):

    def __init__(self, in_channels, n_classes, dropout_rate=0.05, act='relu', n_scalar=1):
        super().__init__()
        if act == 'relu':
            act = nn.ReLU()
        elif act == 'leaky_relu':
            act = nn.LeakyReLU()
        self.seq = nn.Sequential(Conv3DLayer(in_channels, 20, (1, 5, 5), pooling=(1, 2, 2), dropout_rate=dropout_rate, act=act), Conv3DLayer(20, 30, (1, 5, 5), pooling=(1, 2, 2), dropout_rate=dropout_rate, act=act), Conv3DLayer(30, 40, (1, 4, 4), pooling=(1, 2, 2), dropout_rate=dropout_rate, act=act), Conv3DLayer(40, 50, (1, 4, 4), pooling=(1, 2, 2), dropout_rate=dropout_rate, act=act), Conv3DLayer(50, 60, (1, 2, 2), pooling=(1, 2, 2), dropout_rate=dropout_rate, act=act), Conv3DLayer(60, 70, (1, 1, 1), pooling=(1, 2, 2), dropout_rate=dropout_rate, act=act), Conv3DLayer(70, 70, (1, 1, 1), pooling=(1, 1, 1), dropout_rate=dropout_rate, act=act))
        self.adaptavgpool = nn.AdaptiveAvgPool1d(100)
        self.fc = nn.Sequential(nn.Linear(100 + n_scalar, 50), act, nn.Linear(50, 30), act, nn.Linear(30, n_classes))

    def forward(self, x, scal):
        x = self.seq(x)
        x = x.view(x.size()[0], 1, -1)
        x = self.adaptavgpool(x).squeeze(1)
        x = torch.cat((x, scal), 1)
        x = self.fc(x)
        return x


class DenseLayer(nn.Sequential):

    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):

    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features, 1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
            return x


class Bottleneck(nn.Sequential):

    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.add_module('bottleneck', DenseBlock(in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super().forward(x)


class TransitionDown(nn.Sequential):

    def __init__(self, in_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super().forward(x)


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:xy2 + max_height, xy1:xy1 + max_width]


class TransitionUp(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=0, bias=True)

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out


class FCDenseNet(nn.Module):

    def __init__(self, in_channels=3, down_blocks=(5, 5, 5, 5, 5), up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5, growth_rate=16, out_chans_first_conv=48, n_classes=12):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []
        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels, out_channels=out_chans_first_conv, kernel_size=3, stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv
        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += growth_rate * down_blocks[i]
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))
        self.add_module('bottleneck', Bottleneck(cur_channels_count, growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.denseBlocksUp.append(DenseBlock(cur_channels_count, growth_rate, up_blocks[i], upsample=True))
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels
        self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]
        self.denseBlocksUp.append(DenseBlock(cur_channels_count, growth_rate, up_blocks[-1], upsample=False))
        cur_channels_count += growth_rate * up_blocks[-1]
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count, out_channels=n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.firstconv(x)
        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)
        out = self.finalConv(out)
        out = self.softmax(out)
        return out


class PoolingError(Exception):
    pass


class UNet3dLite(nn.Module):
    """(WIP) Re-implementation of the unet3d_lite model from ELEKTRONN2

    See https://github.com/ELEKTRONN/ELEKTRONN2/blob/master/examples/unet3d_lite.py

    Pay attention to shapes: Only spatial input shape (22, 140, 140) is supported.

    fov=[12, 88, 88], offsets=[6, 44, 44], strides=[1 1 1], spatial shape=[10, 52, 52]

    This model is directly compatible with torch.jit.script.
    """

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv3d(1, 32, (1, 3, 3))
        self.conv1 = nn.Conv3d(32, 32, (1, 3, 3))
        self.conv2 = nn.Conv3d(32, 64, (1, 3, 3))
        self.conv3 = nn.Conv3d(64, 64, (1, 3, 3))
        self.conv4 = nn.Conv3d(64, 128, (1, 3, 3))
        self.conv5 = nn.Conv3d(128, 128, (1, 3, 3))
        self.conv6 = nn.Conv3d(128, 256, (3, 3, 3))
        self.conv7 = nn.Conv3d(256, 128, (3, 3, 3))
        self.upconv0 = nn.ConvTranspose3d(128, 512, (1, 2, 2), (1, 2, 2))
        self.mconv0 = nn.Conv3d(640, 256, (1, 3, 3))
        self.mconv1 = nn.Conv3d(256, 64, (1, 3, 3))
        self.upconv1 = nn.ConvTranspose3d(64, 256, (1, 2, 2), (1, 2, 2))
        self.mconv2 = nn.Conv3d(320, 128, (3, 3, 3))
        self.mconv3 = nn.Conv3d(128, 32, (3, 3, 3))
        self.upconv2 = nn.ConvTranspose3d(32, 128, (1, 2, 2), (1, 2, 2))
        self.mconv4 = nn.Conv3d(160, 64, (3, 3, 3))
        self.mconv5 = nn.Conv3d(64, 64, (3, 3, 3))
        self.conv_final = nn.Conv3d(64, 2, 1)

    @staticmethod
    def autocrop(from_down: torch.Tensor, from_up: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        ds = from_down.shape[2:]
        us = from_up.shape[2:]
        from_down = from_down[:, :, (ds[0] - us[0]) // 2:(ds[0] + us[0]) // 2, (ds[1] - us[1]) // 2:(ds[1] + us[1]) // 2, (ds[2] - us[2]) // 2:(ds[2] + us[2]) // 2]
        return from_down, from_up

    @staticmethod
    def down(x, ks=(1, 2, 2)):
        sh = x.shape[2:]
        if any([(s % k != 0) for s, k in zip(sh, ks)]):
            raise PoolingError(f"Can't pool {sh} input by a {ks} kernel. Please adjust the input shape.")
        return F.max_pool3d(x, ks)

    def forward(self, inp):
        conv0 = F.relu(self.conv0(inp))
        conv1 = F.relu(self.conv1(conv0))
        down0 = self.down(conv1)
        conv2 = F.relu(self.conv2(down0))
        conv3 = F.relu(self.conv3(conv2))
        down1 = self.down(conv3)
        conv4 = F.relu(self.conv4(down1))
        conv5 = F.relu(self.conv5(conv4))
        down2 = self.down(conv5)
        conv6 = F.relu(self.conv6(down2))
        conv7 = F.relu(self.conv7(conv6))
        upconv0 = F.relu(self.upconv0(conv7))
        d0, u0 = self.autocrop(conv5, upconv0)
        mrg0 = torch.cat((d0, u0), 1)
        mconv0 = F.relu(self.mconv0(mrg0))
        mconv1 = F.relu(self.mconv1(mconv0))
        upconv1 = F.relu(self.upconv1(mconv1))
        d1, u1 = self.autocrop(conv3, upconv1)
        mrg1 = torch.cat((d1, u1), 1)
        mconv2 = F.relu(self.mconv2(mrg1))
        mconv3 = F.relu(self.mconv3(mconv2))
        upconv2 = F.relu(self.upconv2(mconv3))
        d2, u2 = self.autocrop(conv1, upconv2)
        mrg2 = torch.cat((d2, u2), 1)
        mconv4 = F.relu(self.mconv4(mrg2))
        mconv5 = F.relu(self.mconv5(mconv4))
        out = self.conv_final(mconv5)
        return out


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, True, self.momentum, self.eps)


def ELUCons(relu, nchan):
    if relu:
        return nn.ReLU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):

    def __init__(self, nchan, relu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(relu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        return x


class InputTransition(nn.Module):

    def __init__(self, outChans, relu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, outChans, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.relu1 = ELUCons(relu, outChans)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu1(out)
        return out


def _make_nConv(nchan, depth, relu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, relu))
    return nn.Sequential(*layers)


def passthrough(x, **kwargs):
    return x


class DownTransition(nn.Module):

    def __init__(self, inChans, nConvs, relu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(relu, outChans)
        self.relu2 = ELUCons(relu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, relu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        x = self.do1(down)
        x = self.ops(x)
        x = self.relu2(torch.add(x, down))
        return x


class UpTransition(nn.Module):

    def __init__(self, inChans, outChans, nConvs, relu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(relu, outChans // 2)
        self.relu2 = ELUCons(relu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, relu)

    def forward(self, x, skipx):
        x = self.do1(x)
        skipxdo = self.do2(skipx)
        x = self.relu1(self.bn1(self.up_conv(x)))
        xcat = torch.cat((x, skipxdo), 1)
        x = self.ops(xcat)
        x = self.relu2(torch.add(x, xcat))
        return x


class OutputTransition(nn.Module):

    def __init__(self, inChans, relu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=1)
        self.bn1 = ContBatchNorm3d(2)
        self.relu1 = ELUCons(relu, 2)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        return x


class VNet(nn.Module):

    def __init__(self, relu=True, nll=True, fac=4):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16 // fac, relu)
        self.down_tr32 = DownTransition(16 // fac, 1, relu)
        self.down_tr64 = DownTransition(32 // fac, 2, relu)
        self.down_tr128 = DownTransition(64 // fac, 3, relu, dropout=True)
        self.down_tr256 = DownTransition(128 // fac, 2, relu, dropout=True)
        self.up_tr256 = UpTransition(256 // fac, 256 // fac, 2, relu, dropout=True)
        self.up_tr128 = UpTransition(256 // fac, 128 // fac, 2, relu, dropout=True)
        self.up_tr64 = UpTransition(128 // fac, 64 // fac, 1, relu)
        self.up_tr32 = UpTransition(64 // fac, 32 // fac, 1, relu)
        self.out_tr = OutputTransition(32 // fac, relu, nll)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        del out256
        del out128
        out = self.up_tr128(out, out64)
        del out64
        out = self.up_tr64(out, out32)
        del out32
        out = self.up_tr32(out, out16)
        del out16
        out = self.out_tr(out)
        return out


class Rezero(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.fn(x) * self.g


class Sequential(nn.Module):

    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x) + g(x)
        return x


def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))


def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)


class PermuteToFrom(nn.Module):

    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()
        shape = axial.shape
        *_, t, d = shape
        axial = axial.reshape(-1, t, d)
        axial = self.fn(axial, **kwargs)
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


class AxialPositionalEmbedding(nn.Module):

    def __init__(self, dim, shape, emb_dim_index=1):
        super().__init__()
        parameters = []
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]
        for axial_dim, axial_dim_index in zip(shape, ax_dim_indexes):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[axial_dim_index] = axial_dim
            parameter = nn.Parameter(torch.randn(*shape))
            parameters.append(parameter)
        self.params = nn.ParameterList(parameters)

    def forward(self, x):
        for param in self.params:
            x = x + param
        return x


class SelfAttention(nn.Module):

    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        self.dim_heads = dim // heads if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        q, k, v = self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1)
        b, t, d, h, e = *q.shape, self.heads, self.dim_heads
        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))
        dots = torch.einsum('bie,bje->bij', q, k) * e ** -0.5
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)
        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else emb_dim + total_dimensions
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]
    permutations = []
    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)
    return permutations


class AxialAttention(nn.Module):

    def __init__(self, dim, num_dimensions=2, heads=8, dim_heads=None, dim_index=-1, sum_axial_out=True):
        assert dim % heads == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else dim_index + self.total_dimensions
        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))
        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'
        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))
        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out


class Deterministic(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        if record_rng:
            self.record_rng(*args)
        if not set_rng:
            return self.net(*args, **kwargs)
        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)


class ReversibleBlock(nn.Module):

    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1, y2 = None, None
        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)
        return torch.cat([y1, y2], dim=2)

    def backward_pass(self, y, dy, f_args={}, g_args={}):
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y
        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy
        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)
        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1
            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None
        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)
        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2
            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None
            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)
        return x, dx


class _ReversibleFunction(Function):

    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        kwargs = ctx.kwargs
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None


class ReversibleSequence(nn.Module):

    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for f, g in blocks])

    def forward(self, x, arg_route=(True, True), **kwargs):
        f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
        block_kwargs = {'f_args': f_args, 'g_args': g_args}
        return _ReversibleFunction.apply(x, self.blocks, block_kwargs)


def exists(val):
    return val is not None


class AxialImageTransformer(nn.Module):

    def __init__(self, dim, depth, heads=8, dim_heads=None, dim_index=1, reversible=True, axial_pos_emb_shape=None):
        super().__init__()
        permutations = calculate_permutations(2, dim_index)
        get_ff = lambda : nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.LeakyReLU(inplace=True), nn.Conv2d(dim, dim, 3, padding=1))
        self.pos_emb = AxialPositionalEmbedding(dim, axial_pos_emb_shape, dim_index) if exists(axial_pos_emb_shape) else nn.Identity()
        layers = nn.ModuleList([])
        for _ in range(depth):
            attn_functions = nn.ModuleList([PermuteToFrom(permutation, Rezero(SelfAttention(dim, heads, dim_heads))) for permutation in permutations])
            conv_functions = nn.ModuleList([Rezero(get_ff()), Rezero(get_ff())])
            layers.append(attn_functions)
            layers.append(conv_functions)
        execute_type = ReversibleSequence if reversible else Sequential
        self.layers = execute_type(layers)

    def forward(self, x):
        x = self.pos_emb(x)
        x = torch.cat((x, x), dim=-1)
        x = self.layers(x)
        return torch.stack(x.chunk(2, dim=-1)).mean(dim=0)


class IrreversibleBlock(nn.Module):

    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, x, f_args, g_args):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1 = x1 + self.f(x2, **f_args)
        y2 = x2 + self.g(y1, **g_args)
        return torch.cat([y1, y2], dim=2)


def group_std(x, groups=32, eps=1e-05):
    sh = x.shape
    if x.ndim == 5:
        dims = 2, 3, 4, 5
        n, c, d, h, w = sh
        x = torch.reshape(x, (n, groups, c // groups, d, h, w))
    else:
        dims = 2, 3, 4
        n, c, h, w = sh
        x = torch.reshape(x, (n, groups, c // groups, h, w))
    var = torch.var(x, dim=dims, keepdim=True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), sh)


def instance_std(x, eps=1e-05):
    if x.ndim == 5:
        dims = 2, 3, 4
    else:
        dims = 2, 3
    var = torch.var(x, dim=dims, keepdim=True).expand_as(x)
    if torch.isnan(var).any():
        var = torch.zeros(var.shape)
    return torch.sqrt(var + eps)


class EvoNorm(nn.Module):

    def __init__(self, input, non_linear=True, version='S0', affine=True, momentum=0.9, eps=1e-05, groups=32, training=True, dim=3):
        super().__init__()
        self.non_linear = non_linear
        self.version = version
        self.training = training
        self.momentum = momentum
        self.silu = nn.SiLU()
        self.groups = groups
        self.eps = eps
        if self.version not in ['B0', 'S0']:
            raise ValueError('Invalid EvoNorm version')
        self.insize = input
        self.affine = affine
        self.dim = dim
        if self.dim == 3:
            rs_shape = 1, self.insize, 1, 1, 1
        elif self.dim == 2:
            rs_shape = 1, self.insize, 1, 1
        else:
            raise ValueError('Invalid dim. 2 or 3 expected.')
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(rs_shape))
            self.beta = nn.Parameter(torch.zeros(rs_shape))
            if self.non_linear:
                self.v = nn.Parameter(torch.ones(rs_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
            self.register_buffer('v', None)
        self.register_buffer('running_var', torch.ones(rs_shape))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)

    def _check_input_dim(self, x):
        if x.ndim != self.dim + 2:
            raise ValueError(f'Expected {self.dim + 2}D input but got {x.ndim} input.')

    def forward(self, x):
        self._check_input_dim(x)
        if self.version == 'S0':
            if self.non_linear:
                num = self.silu(x)
                return num / group_std(x, groups=self.groups, eps=self.eps) * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
        if self.version == 'B0':
            if self.training:
                if x.ndim == 5:
                    dims = 0, 2, 3, 4
                else:
                    dims = 0, 2, 3
                var = torch.var(x, dim=dims, unbiased=False, keepdim=True)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var)
            else:
                var = self.running_var
            if self.non_linear:
                den = torch.max((var + self.eps).sqrt(), self.v * x + instance_std(x, eps=self.eps))
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta


class L1BatchNorm(nn.Module):
    """L1-Norm-based Batch Normalization module.

    Use with caution. This code is not extensively tested.

    References:
    - https://arxiv.org/abs/1802.09769
    - https://arxiv.org/abs/1803.01814
    """
    __constants__ = ['l2factor', 'eps', 'momentum']

    def __init__(self, num_features: int, momentum: float=0.9):
        super().__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, num_features))
        self.eps = 1e-05
        self.l2factor = (3.1416 / 2) ** 0.5

    def forward(self, x):
        ndim = x.dim()
        reduce_dims = (0, 2, 3, 4)[:ndim]
        b_sh = (1, x.shape[1], 1, 1, 1)[:ndim]
        if self.training:
            mean = x.mean(dim=reduce_dims, keepdim=True)
            meandiff = x - mean
            absdiff = meandiff.abs()
            l1mean = absdiff.mean(dim=reduce_dims, keepdim=True)
            l1scaled = l1mean * self.l2factor + self.eps
            with torch.no_grad():
                mom = self.momentum
                self.running_mean.mul_(mom).add_(mean.flatten() * (1 - mom))
                self.running_var.mul_(mom).add_(l1scaled.flatten() * (1 - mom))
        else:
            mean = self.running_mean.view(b_sh)
            l1scaled = self.running_var.view(b_sh)
            meandiff = x - mean
        gamma = self.gamma.view(b_sh)
        beta = self.beta.view(b_sh)
        return gamma * meandiff / l1scaled + beta


def l1_group_norm(x, num_groups, weight, bias, eps):
    l2factor = 1.2533
    ndim = x.dim()
    sh = x.shape
    g = num_groups
    n, c = sh[:2]
    grouped_sh = n, g, c // g, *sh[2:]
    grouped = x.view(grouped_sh)
    reduce_dims = (2, 3, 4, 5)[:ndim - 1]
    mean = grouped.mean(dim=reduce_dims, keepdim=True)
    meandiff = grouped - mean
    absdiff = meandiff.abs()
    l1mean = absdiff.mean(dim=reduce_dims, keepdim=True)
    l1scaled = l1mean * l2factor + eps
    normalized = meandiff / l1scaled
    normalized = normalized.view(sh)
    broadcast_sh = (1, c, 1, 1, 1)[:ndim]
    weight = weight.view(broadcast_sh)
    bias = bias.view(broadcast_sh)
    return weight * normalized + bias


class GatherExcite(nn.Module):
    """Gather-Excite module (https://arxiv.org/abs/1810.12348),

    a generalization of the Squeeze-and-Excitation module
    (https://arxiv.org/abs/1709.01507).

    Args:
        channels: Number of input channels (= number of output channels)
        extent: extent factor that determines how much the gather operator
            output is smaller than its input. The special value ``extent=0``
            activates global gathering (so the gathered information has no
            spatial extent).
        param_gather: If ``True``, the gather operator is parametrized
            according to https://arxiv.org/abs/1810.12348.
        param_excite: If ``True``, the excitation operator is parametrized
            according to https://arxiv.org/abs/1810.12348 (also equivalent to
            the original excitation operator proposed in
            https://arxiv.org/abs/1709.01507).
        reduction:  Channel reduction rate of the parametrized excitation
            operator.
        spatial_shape: Spatial shape of the module input. This needs to be
            specified if ``param_gather=0 and extent=0`` (parametrized global
            gathering).
    """

    def __init__(self, channels: int, extent: int=0, param_gather: bool=False, param_excite: bool=True, reduction: int=16, spatial_shape: Optional[Tuple[int, ...]]=None):
        super().__init__()
        if extent == 1:
            raise NotImplementedError("extent == 1 doesn't make sense.")
        if param_gather:
            if extent == 0:
                if spatial_shape is None:
                    raise ValueError('With param_gather=True, extent=0, you will need to specify spatial_shape.')
                self.gather = nn.Sequential(nn.Conv3d(channels, channels, spatial_shape), nn.BatchNorm3d(channels), nn.ReLU())
            else:
                assert extent in [2, 4, 8, 16]
                num_convs = int(torch.log2(torch.tensor(extent, dtype=torch.float32)))
                self.gather = nn.ModuleList([nn.Sequential(nn.Conv3d(channels, channels, 3, stride=2, padding=1), nn.BatchNorm3d(channels), nn.ReLU()) for _ in range(num_convs)])
        elif extent == 0:
            self.gather = nn.AdaptiveAvgPool3d(1)
        else:
            self.gather = nn.AvgPool3d(extent)
        if param_excite:
            self.excite = nn.Sequential(nn.Conv3d(channels, channels // reduction, 1), nn.ReLU(), nn.Conv3d(channels // reduction, channels, 1))
        else:
            self.excite = nn.Identity()
        if extent == 0:
            self.interpolate = nn.Identity()
        else:
            self.interpolate = torch.nn.functional.interpolate

    def forward(self, x):
        y = self.gather(x)
        y = self.excite(y)
        y = torch.sigmoid(self.interpolate(y, x.shape[2:]))
        return x * y


class CombinedLoss(torch.nn.Module):
    """Defines a loss function as a weighted sum of combinable loss criteria.

    Args:
        criteria: List of loss criterion modules that should be combined.
        weight: Weight assigned to the individual loss criteria (in the same
            order as ``criteria``).
        device: The device on which the loss should be computed. This needs
            to be set to the device that the loss arguments are allocated on.
    """

    def __init__(self, criteria: Sequence[torch.nn.Module], weight: Optional[Sequence[float]]=None, device: torch.device=None):
        super().__init__()
        self.criteria = torch.nn.ModuleList(criteria)
        self.device = device
        if weight is None:
            weight = torch.ones(len(criteria))
        else:
            weight = torch.as_tensor(weight, dtype=torch.float32)
            assert weight.shape == (len(criteria),)
        self.register_buffer('weight', weight)

    def forward(self, *args):
        loss = torch.tensor(0.0, device=self.device)
        for crit, weight in zip(self.criteria, self.weight):
            loss += weight * crit(*args)
        return loss


class FocalLoss(torch.nn.Module):
    """Focal Loss (https://arxiv.org/abs/1708.02002)
    
    Expects raw outputs, not softmax probs."""

    def __init__(self, weight=None, gamma=2.0, reduction='mean', ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.nll = torch.nn.NLLLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
        self.log_softmax = torch.nn.LogSoftmax(1)

    def forward(self, output, target):
        log_prob = self.log_softmax(output)
        prob = torch.exp(log_prob)
        return self.nll((1 - prob) ** self.gamma * log_prob, target)


class SoftmaxBCELoss(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bce = torch.nn.BCELoss(*args, **kwargs)

    def forward(self, output, target):
        probs = torch.nn.functional.softmax(output, dim=1)
        return self.bce(probs, target)


def global_average_pooling(inp: torch.Tensor) ->torch.Tensor:
    if inp.ndim == 5:
        return F.adaptive_avg_pool3d(inp, 1)
    elif inp.ndim == 4:
        return F.adaptive_avg_pool2d(inp, 1)
    else:
        raise NotImplementedError


class GAPTripletMarginLoss(nn.TripletMarginLoss):
    """Same as ``torch.nn.TripletMarginLoss``, but applies global average
    pooling to anchor, positive and negative tensors before calculating the
    loss."""

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) ->torch.Tensor:
        return super().forward(global_average_pooling(anchor), global_average_pooling(positive), global_average_pooling(negative))


class MaskedMSELoss(nn.Module):
    """Masked MSE loss where only pixels that are masked are considered.

    Expects an optional binary mask as the third argument.
    If no mask is supplied (``None``), the loss is equivalent to ``torch.nn.MSELoss``."""

    @staticmethod
    def forward(out, target, mask=None):
        if mask is None:
            return F.mse_loss(out, target)
        err = F.mse_loss(out, target, reduction='none')
        err *= mask
        loss = err.sum() / mask.sum()
        return loss


class DistanceWeightedMSELoss(nn.Module):
    """Weighted MSE loss for signed euclidean distance transform targets.

    By setting ``fg_weight`` to a high value, the errors in foreground
    regions are more strongly penalized.
    If ``fg_weight=1``, this loss is equivalent to ``torch.nn.MSELoss``.

    Requires that targets are transformed with
    :py:class:`elektronn3.data.transforms.DistanceTransformTarget`

    Per-pixel weights are assigned on the targets as follows:
    - each foreground pixel is weighted by ``fg_weight``
    - each background pixel is weighted by 1.
    """

    def __init__(self, fg_weight=100.0, mask_borders=40):
        super().__init__()
        self.fg_weight = fg_weight
        self.mask_borders = mask_borders

    def forward(self, output, target):
        mse = nn.functional.mse_loss(output, target, reduction='none')
        with torch.no_grad():
            weight = torch.ones_like(target)
            weight[target <= 0] = self.fg_weight
            if self.mask_borders is not None:
                o = self.mask_borders
                weight[:, :, :o, :o] = 0.0
                weight[:, :, target.shape[-2] - o:, target.shape[-1] - o:] = 0.0
        return torch.mean(weight * mse)


def _channelwise_sum(x: torch.Tensor):
    """Sum-reduce all dimensions of a tensor except dimension 1 (C)"""
    reduce_dims = tuple([0] + list(range(x.dim()))[2:])
    return x.sum(dim=reduce_dims)


def dice_loss(probs, target, weight=1.0, eps=0.0001, smooth=0.0):
    tsh, psh = target.shape, probs.shape
    if tsh == psh:
        onehot_target = target
    elif tsh[0] == psh[0] and tsh[1:] == psh[2:]:
        onehot_target = torch.zeros_like(probs)
        onehot_target.scatter_(1, target.unsqueeze(1), 1)
    else:
        raise ValueError(f'Target shape {target.shape} is not compatible with output shape {probs.shape}.')
    intersection = probs * onehot_target
    numerator = 2 * _channelwise_sum(intersection) + smooth
    denominator = probs + onehot_target
    denominator = _channelwise_sum(denominator) + smooth + eps
    loss_per_channel = 1 - numerator / denominator
    weighted_loss_per_channel = weight * loss_per_channel
    return weighted_loss_per_channel.mean()


class DiceLoss(torch.nn.Module):
    """Generalized Dice Loss, as described in https://arxiv.org/abs/1707.03237.

    Works for n-dimensional data. Assuming that the ``output`` tensor to be
    compared to the ``target`` has the shape (N, C, D, H, W), the ``target``
    can either have the same shape (N, C, D, H, W) (one-hot encoded) or
    (N, D, H, W) (with dense class indices, as in
    ``torch.nn.CrossEntropyLoss``). If the latter shape is detected, the
    ``target`` is automatically internally converted to a one-hot tensor
    for loss calculation.

    Args:
        apply_softmax: If ``True``, a softmax operation is applied to the
            ``output`` tensor before loss calculation. This is necessary if
            your model does not already apply softmax as the last layer.
            If ``False``, ``output`` is assumed to already contain softmax
            probabilities.
        weight: Weight tensor for class-wise loss rescaling.
            Has to be of shape (C,). If ``None``, classes are weighted equally.
        smooth: Smoothing term that is added to both the numerator and the
            denominator of the dice loss formula.
    """

    def __init__(self, apply_softmax: bool=True, weight: Optional[torch.Tensor]=None, smooth: float=0.0):
        super().__init__()
        if apply_softmax:
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.softmax = lambda x: x
        self.dice = dice_loss
        if weight is None:
            weight = torch.tensor(1.0)
        self.register_buffer('weight', weight)
        self.smooth = smooth

    def forward(self, output, target):
        probs = self.softmax(output)
        return self.dice(probs, target, weight=self.weight, smooth=self.smooth)


class FixMatchSegLoss(nn.Module):
    """Self-supervised loss for semi-supervised semantic segmentation training,
    very similar to the :math:`l_u` loss proposed in
    FixMatch (https://arxiv.org/abs/2001.07685).

    The main difference to FixMatch is the kind of augmentations that are used
    for consistency regularization. In FixMatch, so-called
    "strong augmentations" are applied to the (already "weakly augmented")
    inputs. Most of these strong augmentations only work for image-level
    classification.
    In ``FMSegLoss``, only simple, easily reversible geometric augmentations
    are used currently
    (random xy(z) flipping and random xy rotation in 90 degree steps).
    TODO: Add more augmentations

    This loss combines two different well-established semi-supervised learning
    techniques:

    - consistency regularization: consistency (equivariance) against random
      flipping and random rotation augmentatations is enforced
    - pseudo-label training: model argmax predictions are treated as targets
      for a pseudo-supervised cross-entropy training loss
      This only works for settings where argmax makes sense (not suitable for
      regression) and can be disabled with ``enable_psuedo_label=False``.

    Args:
        model: Neural network model to be trained.
        scale: Scalar factor to be multiplied with the loss to adjust its
            magnitude. (If this loss is combined with a standard supervised
            cross entropy, ``scale`` corresponds to the lambda_u hyperparameter
            in FixMatch
        enable_pseudo_label: If ``enable_pseudo_label=True``, the inner loss is
            the cross entropy between the argmax pseudo label tensor computed
            from the weakly augmented input and the softmax model output on the
            strongly augmented input. Since this internally uses
            ``torch.nn.CrossEntropyLoss``, the ``model`` is expected to give
            raw, unsoftmaxed outputs.
            This only works for settings where computing the argmax and softmax
            on the outputs makes sense (so classification, not regression).
            If ``enable_pseudo_label=False``, a mean squared error regression
            loss is computed directly on the difference between the two model
            outputs, without computing or using pseudo-labels.
            In this case, the loss is equivalent to the ``R`` loss proposed in
            "Transformation Consistent Self-ensembling Model for
            Semi-supervised Medical Image Segmentation"
            (https://arxiv.org/abs/1903.00348).
            This non-pseudo-label variant of the loss can also be used for
            pixel-level regression training.
        confidence_thresh: (Only applies if ``enable_pseudo_label=True``.)
            The confidence threshold that determines how
            confident the model has to be in each output element's
            classification for it to contribute to the loss. All output
            elements where none of the softmax class probs exceed this
            threshold are masked out from the loss calculation and the resulting
            loss is set to 0. In the FixMatch paper, this hyperparameter is
            called tau.
        ce_weight: (Only applies if ``enable_pseudo_label=True``.)
            Class weight tensor for the inner cross-entropy loss. Should be
            the same as the weight for the supervised cross-entropy loss.
    """

    def __init__(self, model: nn.Module, scale: float=1.0, enable_pseudo_label: bool=True, confidence_thresh: float=0.9, ce_weight=None):
        super().__init__()
        self.model = model
        self.scale = scale
        self.enable_pseudo_label = enable_pseudo_label
        self.confidence_thresh = confidence_thresh
        if self.enable_pseudo_label:
            self.criterion = nn.CrossEntropyLoss(weight=ce_weight, ignore_index=-100)
        else:
            self.criterion = nn.MSELoss()

    @staticmethod
    def get_random_augmenters(ndim: int) ->Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
        """Produce a pair of functions ``augment, reverse_augment``, where
        the ``augment`` function applies a random augmentation to a torch
        tensor and the ``reverse_augment`` function performs the reverse
        aumentations if applicable (i.e. for geometrical transformations)
        so pixel-level loss calculation is still correct).

        Note that all augmentations are performed on the compute device that
        holds the input, so generally on the GPU.
        """
        k90 = torch.randint(0, 4, ()).item()
        flip_dims_binary = torch.randint(0, 2, (ndim - 2,))
        flip_dims = (torch.nonzero(flip_dims_binary, as_tuple=False).squeeze(1) + 2).tolist()

        @torch.no_grad()
        def augment(x: torch.Tensor) ->torch.Tensor:
            x = torch.rot90(x, +k90, (-1, -2))
            if len(flip_dims) > 0:
                x = torch.flip(x, flip_dims)
            return x

        @torch.no_grad()
        def reverse_augment(x: torch.Tensor) ->torch.Tensor:
            if len(flip_dims) > 0:
                x = torch.flip(x, flip_dims)
            x = torch.rot90(x, -k90, (-1, -2))
            return x
        return augment, reverse_augment

    def forward(self, inp: torch.Tensor) ->torch.Tensor:
        augment, reverse_augment = self.get_random_augmenters(ndim=inp.ndim)
        aug = augment(inp)
        out = self.model(inp)
        aug_out = self.model(aug)
        aug_out_reversed = reverse_augment(aug_out)
        if self.enable_pseudo_label:
            with torch.no_grad():
                out = torch.softmax(out, 1)
                omax, pseudo_label = torch.max(out, dim=1)
                mask = omax < self.confidence_thresh
                pseudo_label[mask] = self.criterion.ignore_index
            loss = self.criterion(aug_out_reversed, pseudo_label)
        else:
            loss = self.criterion(aug_out_reversed, out)
        scaled_loss = self.scale * loss
        return scaled_loss


def norpf_dice_loss(probs, target, weight=1.0, class_weight=1.0):
    tsh, psh = target.shape, probs.shape
    if tsh == psh:
        onehot_target = target
    elif tsh[0] == psh[0] and tsh[1:] == psh[2:]:
        onehot_target = torch.zeros_like(probs)
        onehot_target.scatter_(1, target.unsqueeze(1), 1)
    else:
        raise ValueError(f'Target shape {target.shape} is not compatible with output shape {probs.shape}.')
    if weight.sum() == 0:
        return probs.sum() * 0
    ignore_mask = (1 - onehot_target[0][-1]).view(1, 1, *probs.shape[2:])
    bg_probs = 1 - probs
    bg_target = 1 - onehot_target
    global_weight = (class_weight > 0).type(probs.dtype)
    positive_target_mask = (weight.view(1, -1, 1, 1, 1) * onehot_target)[0][1:-1].sum(dim=0).view(1, 1, *probs.shape[2:])
    weight = weight * global_weight
    dense_weight = weight.view(1, -1, 1, 1, 1)
    target_mask_empty = ((positive_target_mask * ignore_mask).sum(dim=(0, 2, 3, 4)) == 0).type(probs.dtype)
    target_empty = ((onehot_target * ignore_mask).sum(dim=(0, 2, 3, 4)) == 0).type(probs.dtype)
    bg_target_empty = ((bg_target * ignore_mask).sum(dim=(0, 2, 3, 4)) == 0).type(probs.dtype)
    needs_positive_target_mark = (dense_weight.sum() == 0).type(probs.dtype)
    bg_mask = torch.ones_like(bg_probs) * dense_weight + needs_positive_target_mark * positive_target_mask * global_weight.view(1, -1, 1, 1, 1)
    intersection = probs * onehot_target * ignore_mask * dense_weight
    intersection2 = bg_probs * bg_target * ignore_mask * bg_mask
    denominator = (probs + onehot_target) * ignore_mask * dense_weight
    denominator2 = (bg_probs + bg_target) * ignore_mask * bg_mask
    numerator = 2 * class_weight * _channelwise_sum(intersection)
    numerator2 = 2 * _channelwise_sum(intersection2)
    denominator = class_weight * _channelwise_sum(denominator)
    denominator2 = _channelwise_sum(denominator2)
    no_tp = (numerator == 0).type(probs.dtype)
    numerator += 1 - weight
    denominator += 1 - weight
    bg_mask_empty = (bg_mask.sum(dim=(0, 2, 3, 4)) == 0).type(probs.dtype)
    numerator2 *= 1 - bg_mask_empty
    numerator2 += bg_mask_empty
    denominator2 *= 1 - bg_mask_empty
    denominator2 += bg_mask_empty
    numerator *= 1 - target_empty
    numerator += target_empty
    denominator *= 1 - target_empty
    denominator += target_empty
    numerator2 *= 1 - bg_target_empty
    numerator2 += bg_target_empty
    denominator2 *= 1 - bg_target_empty
    denominator2 += bg_target_empty
    if (denominator == 0).sum() > 0 or (denominator2 == 0).sum() > 0:
        None
        IPython.embed()
    loss_per_channel = 1 + no_tp - (numerator / denominator + no_tp * numerator2 / denominator2)
    weighted_loss = loss_per_channel[1:-1].sum() / (class_weight[1:-1] > 0).sum()
    if torch.isnan(weighted_loss).sum() or (weighted_loss > 1).sum():
        None
        IPython.embed()
    return weighted_loss


class NorpfDiceLoss(torch.nn.Module):
    """Generalized Dice Loss, as described in https://arxiv.org/abs/1707.03237,

    Works for n-dimensional data. Assuming that the ``output`` tensor to be
    compared to the ``target`` has the shape (N, C, D, H, W), the ``target``
    can either have the same shape (N, C, D, H, W) (one-hot encoded) or
    (N, D, H, W) (with dense class indices, as in
    ``torch.nn.CrossEntropyLoss``). If the latter shape is detected, the
    ``target`` is automatically internally converted to a one-hot tensor
    for loss calculation.

    Args:
        apply_softmax: If ``True``, a softmax operation is applied to the
            ``output`` tensor before loss calculation. This is necessary if
            your model does not already apply softmax as the last layer.
            If ``False``, ``output`` is assumed to already contain softmax
            probabilities.
        weight: Weight tensor for class-wise loss rescaling.
            Has to be of shape (C,). If ``None``, classes are weighted equally.
    """

    def __init__(self, apply_softmax=True, weight=torch.tensor(1.0), class_weight=torch.tensor(1.0)):
        super().__init__()
        if apply_softmax:
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.softmax = lambda x: x
        self.dice = norpf_dice_loss
        self.register_buffer('weight', weight)
        self.register_buffer('class_weight', class_weight)

    def forward(self, output, target):
        probs = self.softmax(output)
        return self.dice(probs, target, weight=self.weight, class_weight=self.class_weight)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    C = probas.shape[1]
    if probas.dim() == 4:
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    elif probas.dim() == 5:
        probas = probas.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


eps = 0.0001


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / (union + eps)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on num_classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def lovasz_softmax(probas, labels, only_present=False, per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on num_classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), only_present=only_present) for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss


class LovaszLoss(torch.nn.Module):
    """https://arxiv.org/abs/1705.08790"""

    def __init__(self, apply_softmax=True):
        super().__init__()
        if apply_softmax:
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.softmax = lambda x: x
        self.lovasz = lovasz_softmax

    def forward(self, output, target):
        probs = self.softmax(output)
        return self.lovasz(probs, target)


class ACLoss(torch.nn.Module):
    """Active Contour loss
    http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Learning_Active_Contour_Models_for_Medical_Image_Segmentation_CVPR_2019_paper.pdf

    Supports 2D and 3D data, as long as all spatial dimensions have the same
    size and there are only two output channels.

    Modifications:
    - Using mean instead of sum for reductions to avoid size dependency.
    - Instead of the proposed λ loss component weighting (which leads to
      exploding loss magnitudes for high λ values), a relative weight
      ``region_weight`` is used to balance the components:
      ``ACLoss = (1 - region_weight) * length_term + region_weight * region_term``
    """

    def __init__(self, num_classes: int, region_weight: float=0.5):
        assert 0.0 <= region_weight <= 1.0, 'region_weight must be between 0 and 1'
        self.num_classes = num_classes
        self.region_weight = region_weight
        super().__init__()

    @staticmethod
    def get_length(output):
        if output.ndim == 4:
            dy = output[:, :, 1:, :] - output[:, :, :-1, :]
            dx = output[:, :, :, 1:] - output[:, :, :, :-1]
            dy = dy[:, :, 1:, :-2] ** 2
            dx = dx[:, :, :-2, 1:] ** 2
            delta_pred = torch.abs(dy + dx)
        elif output.ndim == 5:
            assert output.shape[3] == output.shape[4], 'All spatial dims must have the same size'
            dz = output[:, :, 1:, :, :] - output[:, :, :-1, :, :]
            dy = output[:, :, :, 1:, :] - output[:, :, :, :-1, :]
            dx = output[:, :, :, :, 1:] - output[:, :, :, :, :-1]
            dz = dz[:, :, 1:, :-2, :-2] ** 2
            dy = dy[:, :, :-2, 1:, :-2] ** 2
            dx = dx[:, :, :-2, :-2, 1:] ** 2
            delta_pred = torch.abs(dz + dy + dx)
        length = torch.mean(torch.sqrt(delta_pred + 1e-06))
        return length

    @staticmethod
    def get_region(output, target):
        region_in = torch.mean(output * (target - 1.0) ** 2.0)
        region_out = torch.mean((1 - output) * target ** 2.0)
        return region_in + region_out

    def forward(self, output, target):
        assert output.shape[2] == output.shape[3], 'All spatial dims must have the same size'
        if target.ndim == output.ndim - 1:
            target = torch.nn.functional.one_hot(target, self.num_classes).transpose(1, -1)
        length_term = self.get_length(output) if self.region_weight < 1.0 else 0.0
        region_term = self.get_region(output, target) if self.region_weight > 0.0 else 0.0
        loss = (1 - self.region_weight) * length_term + self.region_weight * region_term
        return loss


class MixedCombinedLoss(torch.nn.Module):
    """
    Defines a loss function as a weighted sum of combinable loss criteria for multi-class classification with only
    single class ground truths.

    For each voxel, we construct a 2 channel output after the softmax:
     channel 0: background (actual background + all but one classes) = (1-channel 1)
     channel 1: foreground (the one class to which the target corresponds)

    Args:
        class_weight: a manual rescaling weight given to each
            class.
        criteria: List of loss criterion modules that should be combined.
        criteria_weight: Weight assigned to the individual loss criteria (in the same
            order as ``criteria``).
        device: The device on which the loss should be computed. This needs
            to be set to the device that the loss arguments are allocated on.
        eps:
    """

    def __init__(self, class_weight, criteria, criteria_weight, device, eps=1e-10, **kwargs):
        super(MixedCombinedLoss, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.class_weight = class_weight
        self.criteria = torch.nn.ModuleList(criteria)
        self.device = device
        self.eps = eps
        if criteria_weight is None:
            weight = torch.ones(len(criteria))
        else:
            weight = torch.as_tensor(criteria_weight, dtype=torch.float32)
            assert weight.shape == (len(criteria),)
        self.register_buffer('weight', weight)

    def forward(self, output_direct, target, target_class):
        assert all([(len(torch.unique(target_sample[0])) <= 2) for target_sample in target])
        modified_target = torch.zeros_like(target)
        modified_target[target != 0] = 1
        logit_max = output_direct.max(axis=1)[0].unsqueeze(1)
        output_shifted = output_direct - logit_max
        softmax_output = self.softmax(output_shifted)
        softmax_output = (1 - self.eps) * softmax_output + self.eps
        softmax_output = softmax_output[range(softmax_output.shape[0]), target_class].unsqueeze(1)
        softmax_output = torch.cat([1 - softmax_output, softmax_output], dim=1)
        exp_output = output_shifted.exp()
        exp_output_sum = exp_output.sum(axis=1).unsqueeze(1)
        exp_output_sum_log = exp_output_sum.log()
        log_softmax_i = output_shifted - exp_output_sum_log
        num_classes = output_direct.shape[1]
        idx = [(np.arange(num_classes) != i) for i in range(num_classes)]
        exp_output_sum_minus_i = torch.stack([exp_output[:, idx[k]].sum(axis=1) for k in range(num_classes)], dim=1)
        log_softmax_minus_i = exp_output_sum_minus_i.log() - exp_output_sum_log
        log_softmax_i_output = log_softmax_i[range(softmax_output.shape[0]), target_class].unsqueeze(1)
        log_softmax_minus_i_output = log_softmax_minus_i[range(softmax_output.shape[0]), target_class].unsqueeze(1)
        log_softmax_output = torch.cat([log_softmax_minus_i_output, log_softmax_i_output], dim=1)
        loss = torch.tensor(0.0, device=softmax_output.device)
        for crit, crit_weight in zip(self.criteria, self.weight):
            for i in range(softmax_output.shape[0]):
                if isinstance(crit, torch.nn.NLLLoss):
                    crit_loss = crit(log_softmax_output[i].unsqueeze(0), modified_target[i].unsqueeze(0)).mean()
                elif isinstance(crit, DiceLoss):
                    crit_loss = crit(softmax_output[i].unsqueeze(0), modified_target[i].unsqueeze(0))
                else:
                    raise NotImplementedError()
                loss += crit_loss * crit_weight * self.class_weight[target_class[i]]
        return loss


class StableBCELoss(torch.nn.modules.Module):

    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class FWS(nn.Module):
    """Kind of like weight standardization, but changes weights in place"""

    def __init__(self, layer: nn.Module, learnable_gain: bool=True, const_eval: bool=False, eps: float=0.0001):
        super().__init__()
        self.layer = layer
        self.const_eval = const_eval
        self.vmdims: Tuple[int, ...] = tuple(range(1, self.layer.weight.ndim))
        if learnable_gain:
            self.gain = nn.Parameter(torch.ones(self.layer.weight.shape[0], requires_grad=True))
        else:
            self.register_buffer('gain', torch.ones(self.layer.weight.shape[0]))
        self.register_buffer('fan_in', torch.prod(torch.tensor(self.layer.weight.shape)))
        self.register_buffer('eps', torch.tensor(eps))

    def standardize_weight(self):
        var, mean = torch.var_mean(self.layer.weight, dim=self.vmdims, keepdims=True)
        scale = torch.rsqrt(torch.max(var * self.fan_in, self.eps)) * self.gain.view_as(var)
        shift = mean * scale
        with torch.no_grad():
            self.layer.weight.mul_(scale).sub_(shift)

    def forward(self, *args, **kwargs):
        if self.training or not self.const_eval:
            self.standardize_weight()
        return self.layer(*args, **kwargs)


class WSConv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        nn.init.kaiming_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.weight.shape[0], requires_grad=True))

    def standardize_weight(self, eps):
        var, mean = torch.var_mean(self.weight, dim=(1, 2, 3, 4), keepdims=True)
        fan_in = torch.prod(torch.tensor(self.weight.shape))
        scale = torch.rsqrt(torch.max(var * fan_in, torch.tensor(eps))) * self.gain.view_as(var)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, input, eps=0.0001):
        weight = self.standardize_weight(eps)
        return F.conv3d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class WSConvTranspose3d(nn.ConvTranspose3d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, output_padding=0, groups: int=1, bias: bool=True, dilation: int=1, padding_mode: str='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode)
        nn.init.kaiming_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.weight.size(0), requires_grad=True))

    def standardize_weight(self, eps):
        var, mean = torch.var_mean(self.weight, dim=(1, 2, 3, 4), keepdims=True)
        fan_in = torch.prod(torch.tensor(self.weight.shape))
        scale = torch.rsqrt(torch.max(var * fan_in, torch.tensor(eps))) * self.gain.view_as(var)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, input: Tensor, output_size: Optional[List[int]]=None, eps: float=0.0001) ->Tensor:
        weight = self.standardize_weight(eps)
        return F.conv_transpose2d(input, weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)


class WSConv1d(nn.Conv1d):
    """Applies a 1D convolution over an input signal composed of several input
    planes.
    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\\text{in}}, L)` and output :math:`(N, C_{\\text{out}}, L_{\\text{out}})` can be
    precisely described as:
    .. math::
        \\text{out}(N_i, C_{\\text{out}_j}) = \\text{bias}(C_{\\text{out}_j}) +
        \\sum_{k = 0}^{C_{in} - 1} \\text{weight}(C_{\\text{out}_j}, k)
        \\star \\text{input}(N_i, k)
    where :math:`\\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.
    * :attr:`padding` controls the amount of implicit zero-paddings on both sides
      for :attr:`padding` number of points.
    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.
    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,
        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters,
          of size
          :math:`\\left\\lfloor\\frac{out\\_channels}{in\\_channels}\\right\\rfloor`.
    Note:
        Depending of the size of your kernel, several (of the last)
        columns of the input might be lost, because it is a valid
        `cross-correlation`_, and not a full `cross-correlation`_.
        It is up to the user to add proper padding.
    Note:
        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also termed in
        literature as depthwise convolution.
        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
        a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
        :math:`(C_\\text{in}=C_{in}, C_\\text{out}=C_{in} \\times K, ..., \\text{groups}=C_{in})`.
    Note:
        In some circumstances when using the CUDA backend with CuDNN, this operator
        may select a nondeterministic algorithm to increase performance. If this is
        undesirable, you can try to make the operation deterministic (potentially at
        a performance cost) by setting ``torch.backends.cudnn.deterministic =
        True``.
        Please see the notes on :doc:`/notes/randomness` for background.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where
          .. math::
              L_{out} = \\left\\lfloor\\frac{L_{in} + 2 \\times \\text{padding} - \\text{dilation}
                        \\times (\\text{kernel\\_size} - 1) - 1}{\\text{stride}} + 1\\right\\rfloor
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\\text{out\\_channels},
            \\frac{\\text{in\\_channels}}{\\text{groups}}, \\text{kernel\\_size})`.
            The values of these weights are sampled from
            :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where
            :math:`k = \\frac{groups}{C_\\text{in} * \\text{kernel\\_size}}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``, then the values of these weights are
            sampled from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where
            :math:`k = \\frac{groups}{C_\\text{in} * \\text{kernel\\_size}}`
    Examples::
        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)
    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation
    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        nn.init.kaiming_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.weight.size()[0], requires_grad=True))

    def standardize_weight(self, eps):
        var, mean = torch.var_mean(self.weight, dim=(1, 2), keepdims=True)
        fan_in = torch.prod(torch.tensor(self.weight.shape))
        scale = torch.rsqrt(torch.max(var * fan_in, torch.tensor(eps))) * self.gain.view_as(var)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, input, eps=0.0001):
        weight = self.standardize_weight(eps)
        return F.conv1d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class WSConv2d(nn.Conv2d):
    "Applies a 2D convolution over an input signal composed of several input\n    planes after weight normalization/standardization.\n    Reference: https://github.com/deepmind/deepmind-research/blob/master/nfnets/base.py#L121\n    In the simplest case, the output value of the layer with input size\n    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`\n    can be precisely described as:\n    .. math::\n        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +\n        \\sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \\star \text{input}(N_i, k)\n    where :math:`\\star` is the valid 2D `cross-correlation`_ operator,\n    :math:`N` is a batch size, :math:`C` denotes a number of channels,\n    :math:`H` is a height of input planes in pixels, and :math:`W` is\n    width in pixels.\n    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.\n    * :attr:`stride` controls the stride for the cross-correlation, a single\n      number or a tuple.\n    * :attr:`padding` controls the amount of implicit zero-paddings on both\n      sides for :attr:`padding` number of points for each dimension.\n    * :attr:`dilation` controls the spacing between the kernel points; also\n      known as the à trous algorithm. It is harder to describe, but this `link`_\n      has a nice visualization of what :attr:`dilation` does.\n    * :attr:`groups` controls the connections between inputs and outputs.\n      :attr:`in_channels` and :attr:`out_channels` must both be divisible by\n      :attr:`groups`. For example,\n        * At groups=1, all inputs are convolved to all outputs.\n        * At groups=2, the operation becomes equivalent to having two conv\n          layers side by side, each seeing half the input channels,\n          and producing half the output channels, and both subsequently\n          concatenated.\n        * At groups= :attr:`in_channels`, each input channel is convolved with\n          its own set of filters, of size:\n          :math:`\\left\\lfloor\x0crac{out\\_channels}{in\\_channels}\right\rfloor`.\n    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:\n        - a single ``int`` -- in which case the same value is used for the height and width dimension\n        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,\n          and the second `int` for the width dimension\n    Note:\n         Depending of the size of your kernel, several (of the last)\n         columns of the input might be lost, because it is a valid `cross-correlation`_,\n         and not a full `cross-correlation`_.\n         It is up to the user to add proper padding.\n    Note:\n        When `groups == in_channels` and `out_channels == K * in_channels`,\n        where `K` is a positive integer, this operation is also termed in\n        literature as depthwise convolution.\n        In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`,\n        a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments\n        :math:`(in\\_channels=C_{in}, out\\_channels=C_{in} \times K, ..., groups=C_{in})`.\n    Note:\n        In some circumstances when using the CUDA backend with CuDNN, this operator\n        may select a nondeterministic algorithm to increase performance. If this is\n        undesirable, you can try to make the operation deterministic (potentially at\n        a performance cost) by setting ``torch.backends.cudnn.deterministic =\n        True``.\n        Please see the notes on :doc:`/notes/randomness` for background.\n    Args:\n        in_channels (int): Number of channels in the input image\n        out_channels (int): Number of channels produced by the convolution\n        kernel_size (int or tuple): Size of the convolving kernel\n        stride (int or tuple, optional): Stride of the convolution. Default: 1\n        padding (int or tuple, optional): Zero-padding added to both sides of\n            the input. Default: 0\n        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,\n            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``\n        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1\n        groups (int, optional): Number of blocked connections from input\n            channels to output channels. Default: 1\n        bias (bool, optional): If ``True``, adds a learnable bias to the\n            output. Default: ``True``\n    Shape:\n        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`\n        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where\n          .. math::\n              H_{out} = \\left\\lfloor\x0crac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]\n                        \times (\text{kernel\\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor\n          .. math::\n              W_{out} = \\left\\lfloor\x0crac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]\n                        \times (\text{kernel\\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor\n    Attributes:\n        weight (Tensor): the learnable weights of the module of shape\n            :math:`(\text{out\\_channels}, \x0crac{\text{in\\_channels}}{\text{groups}},`\n            :math:`\text{kernel\\_size[0]}, \text{kernel\\_size[1]})`.\n            The values of these weights are sampled from\n            :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where\n            :math:`k = \x0crac{groups}{C_\text{in} * \\prod_{i=0}^{1}\text{kernel\\_size}[i]}`\n        bias (Tensor):   the learnable bias of the module of shape\n            (out_channels). If :attr:`bias` is ``True``,\n            then the values of these weights are\n            sampled from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where\n            :math:`k = \x0crac{groups}{C_\text{in} * \\prod_{i=0}^{1}\text{kernel\\_size}[i]}`\n    Examples:\n        >>> # With square kernels and equal stride\n        >>> m = WSConv2d(16, 33, 3, stride=2)\n        >>> # non-square kernels and unequal stride and with padding\n        >>> m = WSConv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))\n        >>> # non-square kernels and unequal stride and with padding and dilation\n        >>> m = WSConv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))\n        >>> input = torch.randn(20, 16, 50, 100)\n        >>> output = m(input)\n    .. _cross-correlation:\n        https://en.wikipedia.org/wiki/Cross-correlation\n    .. _link:\n        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md\n    "

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        nn.init.kaiming_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.weight.size(0), requires_grad=True))

    def standardize_weight(self, eps):
        var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdims=True)
        fan_in = torch.prod(torch.tensor(self.weight.shape[0:]))
        scale = torch.rsqrt(torch.max(var * fan_in, torch.tensor(eps))) * self.gain.view_as(var)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, input, eps=0.0001):
        weight = self.standardize_weight(eps)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class WSConvTranspose2d(nn.ConvTranspose2d):
    'Applies a 2D transposed convolution operator over an input image\n    composed of several input planes after weight normalization/standardization.\n    This module can be seen as the gradient of Conv2d with respect to its input.\n    It is also known as a fractionally-strided convolution or\n    a deconvolution (although it is not an actual deconvolution operation).\n    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.\n    * :attr:`stride` controls the stride for the cross-correlation.\n    * :attr:`padding` controls the amount of implicit zero-paddings on both\n      sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note\n      below for details.\n    * :attr:`output_padding` controls the additional size added to one side\n      of the output shape. See note below for details.\n    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.\n      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.\n    * :attr:`groups` controls the connections between inputs and outputs.\n      :attr:`in_channels` and :attr:`out_channels` must both be divisible by\n      :attr:`groups`. For example,\n        * At groups=1, all inputs are convolved to all outputs.\n        * At groups=2, the operation becomes equivalent to having two conv\n          layers side by side, each seeing half the input channels,\n          and producing half the output channels, and both subsequently\n          concatenated.\n        * At groups= :attr:`in_channels`, each input channel is convolved with\n          its own set of filters (of size\n          :math:`\\left\\lfloor\x0crac{out\\_channels}{in\\_channels}\right\rfloor`).\n    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`\n    can either be:\n        - a single ``int`` -- in which case the same value is used for the height and width dimensions\n        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,\n          and the second `int` for the width dimension\n    .. note::\n         Depending of the size of your kernel, several (of the last)\n         columns of the input might be lost, because it is a valid `cross-correlation`_,\n         and not a full `cross-correlation`_.\n         It is up to the user to add proper padding.\n    Note:\n        The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``\n        amount of zero padding to both sizes of the input. This is set so that\n        when a :class:`~torch.nn.Conv2d` and a :class:`~torch.nn.ConvTranspose2d`\n        are initialized with same parameters, they are inverses of each other in\n        regard to the input and output shapes. However, when ``stride > 1``,\n        :class:`~torch.nn.Conv2d` maps multiple input shapes to the same output\n        shape. :attr:`output_padding` is provided to resolve this ambiguity by\n        effectively increasing the calculated output shape on one side. Note\n        that :attr:`output_padding` is only used to find output shape, but does\n        not actually add zero-padding to output.\n    Note:\n        In some circumstances when using the CUDA backend with CuDNN, this operator\n        may select a nondeterministic algorithm to increase performance. If this is\n        undesirable, you can try to make the operation deterministic (potentially at\n        a performance cost) by setting ``torch.backends.cudnn.deterministic =\n        True``.\n        Please see the notes on :doc:`/notes/randomness` for background.\n    Args:\n        in_channels (int): Number of channels in the input image\n        out_channels (int): Number of channels produced by the convolution\n        kernel_size (int or tuple): Size of the convolving kernel\n        stride (int or tuple, optional): Stride of the convolution. Default: 1\n        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding\n            will be added to both sides of each dimension in the input. Default: 0\n        output_padding (int or tuple, optional): Additional size added to one side\n            of each dimension in the output shape. Default: 0\n        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1\n        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``\n        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1\n    Shape:\n        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`\n        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where\n        .. math::\n              H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]\n                        \times (\text{kernel\\_size}[0] - 1) + \text{output\\_padding}[0] + 1\n        .. math::\n              W_{out} = (W_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]\n                        \times (\text{kernel\\_size}[1] - 1) + \text{output\\_padding}[1] + 1\n    Attributes:\n        weight (Tensor): the learnable weights of the module of shape\n                         :math:`(\text{in\\_channels}, \x0crac{\text{out\\_channels}}{\text{groups}},`\n                         :math:`\text{kernel\\_size[0]}, \text{kernel\\_size[1]})`.\n                         The values of these weights are sampled from\n                         :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where\n                         :math:`k = \x0crac{groups}{C_\text{out} * \\prod_{i=0}^{1}\text{kernel\\_size}[i]}`\n        bias (Tensor):   the learnable bias of the module of shape (out_channels)\n                         If :attr:`bias` is ``True``, then the values of these weights are\n                         sampled from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where\n                         :math:`k = \x0crac{groups}{C_\text{out} * \\prod_{i=0}^{1}\text{kernel\\_size}[i]}`\n    Examples::\n        >>> # With square kernels and equal stride\n        >>> m = WSConvTranspose2d(16, 33, 3, stride=2)\n        >>> # non-square kernels and unequal stride and with padding\n        >>> m = WSConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))\n        >>> input = torch.randn(20, 16, 50, 100)\n        >>> output = m(input)\n        >>> # exact output size can be also specified as an argument\n        >>> input = torch.randn(1, 16, 12, 12)\n        >>> downsample = WSConv2d(16, 16, 3, stride=2, padding=1)\n        >>> upsample = WSConvTranspose2d(16, 16, 3, stride=2, padding=1)\n        >>> h = downsample(input)\n        >>> h.size()\n        torch.Size([1, 16, 6, 6])\n        >>> output = upsample(h, output_size=input.size())\n        >>> output.size()\n        torch.Size([1, 16, 12, 12])\n    .. _cross-correlation:\n        https://en.wikipedia.org/wiki/Cross-correlation\n    .. _link:\n        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md\n    '

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, output_padding=0, groups: int=1, bias: bool=True, dilation: int=1, padding_mode: str='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode)
        nn.init.kaiming_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.weight.size(0), requires_grad=True))

    def standardize_weight(self, eps):
        var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdims=True)
        fan_in = torch.prod(torch.tensor(self.weight.shape[0:]))
        scale = torch.rsqrt(torch.max(var * fan_in, torch.tensor(eps))) * self.gain.view_as(var)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, input: Tensor, output_size: Optional[List[int]]=None, eps: float=0.0001) ->Tensor:
        weight = self.standardize_weight(eps)
        return F.conv_transpose2d(input, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ACLoss,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Argmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AxialPositionalEmbedding,
     lambda: ([], {'dim': 4, 'shape': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'in_channels': 4, 'growth_rate': 4, 'n_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ContBatchNorm3d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Conv3DLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DenseBlock,
     lambda: ([], {'in_channels': 4, 'growth_rate': 4, 'n_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DenseLayer,
     lambda: ([], {'in_channels': 4, 'growth_rate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Deterministic,
     lambda: ([], {'net': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (DiceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DistanceWeightedMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownTransition,
     lambda: ([], {'inChans': 4, 'nConvs': 4, 'relu': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (DummyAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Extended3DNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
    (FCDenseNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (FixMatchSegLoss,
     lambda: ([], {'model': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GAPTripletMarginLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (InputTransition,
     lambda: ([], {'outChans': 4, 'relu': 4}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
    (L1BatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LUConv,
     lambda: ([], {'nchan': 4, 'relu': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (LovaszLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([64, 4]), torch.rand([16, 4])], {}),
     False),
    (MSDNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
    (MaskedMSELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (N3DNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
    (OutputTransition,
     lambda: ([], {'inChans': 4, 'relu': 4, 'nll': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (ResizeConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Rezero,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelfAttention,
     lambda: ([], {'dim': 4, 'heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Simple3DNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
    (SoftmaxBCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (StableBCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (StackedConv2Scalar,
     lambda: ([], {'in_channels': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 64, 128, 128])], {}),
     True),
    (TransitionDown,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TransitionUp,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (UNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
    (VNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     False),
    (WSConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (WSConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (WSConv3d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (WSConvTranspose2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (fcn16s,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
    (fcn32s,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
    (fcn8s,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
]

class Test_ELEKTRONN_elektronn3(_paritybench_base):
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

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

    def test_040(self):
        self._check(*TESTCASES[40])

    def test_041(self):
        self._check(*TESTCASES[41])

    def test_042(self):
        self._check(*TESTCASES[42])

    def test_043(self):
        self._check(*TESTCASES[43])

    def test_044(self):
        self._check(*TESTCASES[44])

    def test_045(self):
        self._check(*TESTCASES[45])

