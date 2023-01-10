import sys
_module = sys.modules[__name__]
del sys
core = _module
base = _module
model = _module
test = _module
train = _module
warmup = _module
main = _module
tools = _module
data_loader = _module
loader = _module
reid_dataset = _module
reid_samples = _module
evaluation = _module
classification = _module
sysu_mm01 = _module
python_api = _module
sysu_mm01_python = _module
evaluate_sysymm01 = _module
logger = _module
loss2 = _module
meter = _module
metric = _module
transforms2 = _module
utils = _module

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


import copy


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import itertools


import torchvision


import numpy as np


import scipy.io as sio


from torchvision.utils import save_image


import torch.utils.data as data


import torchvision.transforms as transforms


import random


import time


class ResidualBlock(nn.Module):
    """Residual Block with Instance Normalization"""

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True))

    def forward(self, x):
        return self.model(x) + x


class Generator(nn.Module):
    """Generator with Down sampling, Several ResBlocks and Up sampling.
       Down/Up Samplings are used for less computation.
    """

    def __init__(self, conv_dim, layer_num):
        super(Generator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=3, out_channels=conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        current_dims = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(current_dims, current_dims * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(current_dims * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            current_dims *= 2
        for i in range(layer_num):
            layers.append(ResidualBlock(current_dims, current_dims))
        for i in range(2):
            layers.append(nn.ConvTranspose2d(current_dims, current_dims // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(current_dims // 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            current_dims = current_dims // 2
        layers.append(nn.Conv2d(current_dims, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """Discriminator with PatchGAN"""

    def __init__(self, image_size, conv_dim, layer_num):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        current_dim = conv_dim
        for i in range(layer_num):
            layers.append(nn.Conv2d(current_dim, current_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(current_dim * 2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_dim *= 2
        self.model = nn.Sequential(*layers)
        self.conv_src = nn.Conv2d(current_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.model(x)
        out_src = self.conv_src(x)
        return out_src


class ConditionalDiscriminator(nn.Module):
    """Discriminator with PatchGAN"""

    def __init__(self, conv_dim):
        super(ConditionalDiscriminator, self).__init__()
        image_convs = []
        image_convs.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        image_convs.append(nn.LeakyReLU(0.2, inplace=True))
        current_dim = conv_dim
        for i in range(2):
            image_convs.append(nn.Conv2d(current_dim, current_dim * 2, kernel_size=4, stride=2, padding=1))
            image_convs.append(nn.InstanceNorm2d(current_dim * 2))
            image_convs.append(nn.LeakyReLU(0.2, inplace=True))
            current_dim *= 2
        self.image_convs = nn.Sequential(*image_convs)
        feature_convs = []
        feature_convs.append(nn.Conv2d(2048, conv_dim, kernel_size=1, stride=1, padding=0))
        feature_convs.append(nn.LeakyReLU(0.2, inplace=True))
        self.feature_convs = nn.Sequential(*feature_convs)
        dis_convs = []
        dis_convs.append(nn.Conv2d(current_dim + conv_dim, current_dim * 2, kernel_size=4, stride=2, padding=1))
        dis_convs.append(nn.InstanceNorm2d(current_dim * 2))
        dis_convs.append(nn.LeakyReLU(0.2, inplace=True))
        current_dim *= 2
        dis_convs.append(nn.Conv2d(current_dim, current_dim * 2, kernel_size=4, stride=2, padding=1))
        dis_convs.append(nn.InstanceNorm2d(current_dim * 2))
        dis_convs.append(nn.LeakyReLU(0.2, inplace=True))
        current_dim *= 2
        self.dis_convs = nn.Sequential(*dis_convs)
        self.conv_src = nn.Conv2d(current_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, images, features):
        images = self.image_convs(images)
        features = self.feature_convs(features)
        features = F.interpolate(features, [16, 16], mode='bilinear')
        x = torch.cat([images, features], dim=1)
        x = self.dis_convs(x)
        out_src = self.conv_src(x)
        return out_src


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class BottleClassifier(nn.Module):

    def __init__(self, in_dim, out_dim, relu=True, dropout=True, bottle_dim=512):
        super(BottleClassifier, self).__init__()
        bottle = [nn.Linear(in_dim, bottle_dim)]
        bottle += [nn.BatchNorm1d(bottle_dim)]
        if relu:
            bottle += [nn.LeakyReLU(0.1)]
        if dropout:
            bottle += [nn.Dropout(p=0.5)]
        bottle = nn.Sequential(*bottle)
        bottle.apply(weights_init_kaiming)
        self.bottle = bottle
        classifier = [nn.Linear(bottle_dim, out_dim)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.bottle(x)
        x = self.classifier(x)
        return x


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = 1, 1
        resnet.layer4[0].conv2.stride = 1, 1
        self.resnet_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

    def forward(self, x):
        return self.resnet_conv(x)


class Embeder(nn.Module):

    def __init__(self, part_num, class_num):
        super(Embeder, self).__init__()
        self.part_num = part_num
        self.class_num = class_num
        avgpool = nn.AdaptiveAvgPool2d((self.part_num, 1))
        dropout = nn.Dropout(p=0.5)
        self.pool_c = nn.Sequential(avgpool, dropout)
        self.pool_e = nn.Sequential(avgpool)
        for i in range(part_num):
            name = 'classifier' + str(i)
            setattr(self, name, BottleClassifier(2048, self.class_num, relu=True, dropout=False, bottle_dim=256))
            name = 'embedder' + str(i)
            setattr(self, name, nn.Linear(2048, 256))

    def forward(self, features):
        features_c = torch.squeeze(self.pool_c(features))
        features_e = torch.squeeze(self.pool_e(features))
        logits_list = []
        for i in range(self.part_num):
            if self.part_num == 1:
                features_i = features_c
            else:
                features_i = torch.squeeze(features_c[:, :, i])
            classifier_i = getattr(self, 'classifier' + str(i))
            logits_i = classifier_i(features_i)
            logits_list.append(logits_i)
        embedding_list = []
        for i in range(self.part_num):
            if self.part_num == 1:
                features_i = features_e
            else:
                features_i = torch.squeeze(features_e[:, :, i])
            embedder_i = getattr(self, 'embedder' + str(i))
            embedding_i = embedder_i(features_i)
            embedding_list.append(embedding_i)
        return features_c, features_e, logits_list, embedding_list


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BottleClassifier,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Discriminator,
     lambda: ([], {'image_size': 4, 'conv_dim': 4, 'layer_num': 1}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Encoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Generator,
     lambda: ([], {'conv_dim': 4, 'layer_num': 1}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ResidualBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_wangguanan_AlignGAN(_paritybench_base):
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

