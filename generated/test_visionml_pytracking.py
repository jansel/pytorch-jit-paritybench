import sys
_module = sys.modules[__name__]
del sys
ltr = _module
actors = _module
base_actor = _module
bbreg = _module
segmentation = _module
tracking = _module
admin = _module
environment = _module
loading = _module
model_constructor = _module
multigpu = _module
settings = _module
stats = _module
tensorboard = _module
data = _module
bounding_box_utils = _module
image_loader = _module
loader = _module
processing = _module
processing_utils = _module
sampler = _module
transforms = _module
dataset = _module
base_image_dataset = _module
base_video_dataset = _module
coco = _module
coco_seq = _module
davis = _module
ecssd = _module
got10k = _module
got10kvos = _module
hku_is = _module
imagenetvid = _module
lasot = _module
lasot_candidate_matching = _module
lasotvos = _module
lvis = _module
msra10k = _module
sbd = _module
synthetic_video = _module
synthetic_video_blend = _module
tracking_net = _module
vos_base = _module
youtubevos = _module
models = _module
backbone = _module
base = _module
mobilenetv3 = _module
resnet = _module
resnet18_vggm = _module
resnet_mrcnn = _module
atom = _module
atom_iou_net = _module
kys = _module
conv_gru = _module
cost_volume = _module
predictor_wrapper = _module
response_predictor = _module
utils = _module
layers = _module
activation = _module
blocks = _module
distance = _module
filter = _module
normalization = _module
transform = _module
loss = _module
bbr_loss = _module
kl_regression = _module
lovasz_loss = _module
segmentation = _module
target_candidate_matching_loss = _module
target_classification = _module
lwl = _module
decoder = _module
initializer = _module
label_encoder = _module
linear_filter = _module
loss_residual_modules = _module
lwl_box_net = _module
lwl_net = _module
sta_net = _module
utils = _module
meta = _module
steepestdescent = _module
rts = _module
decoder = _module
initializer = _module
label_encoder = _module
learners_fusion = _module
linear_filter = _module
loss_residual_modules = _module
rts_net = _module
utils = _module
target_candidate_matching = _module
superglue = _module
target_candidate_matching = _module
target_classifier = _module
features = _module
initializer = _module
linear_filter = _module
optimizer = _module
residual_modules = _module
dimpnet = _module
kysnet = _module
tompnet = _module
transformer = _module
filter_predictor = _module
heads = _module
position_encoding = _module
transformer = _module
run_training = _module
train_settings = _module
atom = _module
atom_gmm_sampl = _module
atom_paper = _module
atom_prob_ml = _module
dimp = _module
dimp18 = _module
dimp50 = _module
prdimp18 = _module
prdimp50 = _module
super_dimp = _module
super_dimp_simple = _module
keep_track = _module
keep_track = _module
kys = _module
lwl_boxinit = _module
lwl_stage1 = _module
lwl_stage2 = _module
rts50 = _module
tomp = _module
tomp101 = _module
tomp50 = _module
trainers = _module
base_trainer = _module
ltr_trainer = _module
vot = _module
pytracking = _module
analysis = _module
evaluate_vos = _module
extract_results = _module
playback_results = _module
plot_results = _module
vos_utils = _module
evaluation = _module
avistdataset = _module
datasets = _module
got10kdataset = _module
lasotdataset = _module
lasotextensionsubsetdataset = _module
mobifacedataset = _module
multi_object_wrapper = _module
nfsdataset = _module
otbdataset = _module
oxuvadataset = _module
running = _module
tpldataset = _module
tracker = _module
trackingnetdataset = _module
uavdataset = _module
vot2020 = _module
votdataset = _module
experiments = _module
myexperiments = _module
augmentation = _module
color = _module
deep = _module
extractor = _module
featurebase = _module
net_wrappers = _module
preprocessing = _module
util = _module
libs = _module
complex = _module
dcf = _module
fourier = _module
operation = _module
optimization = _module
tensordict = _module
tensorlist = _module
parameter = _module
atom_gmm_sampl = _module
atom_prob_ml = _module
default = _module
default_vot = _module
multiscale_no_iounet = _module
dimp18_vot18 = _module
dimp50_vot18 = _module
dimp50_vot19 = _module
prdimp50_vot18 = _module
dimp_simple = _module
eco = _module
default = _module
mobile3 = _module
default_fast = _module
lwl_ytvos = _module
run_experiment = _module
run_tracker = _module
run_video = _module
run_vot = _module
run_webcam = _module
atom = _module
optim = _module
basetracker = _module
dimp = _module
dimp_simple = _module
eco = _module
optim = _module
candidates = _module
keep_track = _module
kys = _module
lwl = _module
clf_branch = _module
rts = _module
sta_helper = _module
tomp = _module
util_scripts = _module
create_distractor_dataset = _module
download_results = _module
pack_got10k_results = _module
pack_trackingnet_results = _module
convert_vot_anno_to_rect = _module
load_text = _module
params = _module
plotting = _module
visdom = _module

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


import numpy as np


import inspect


from collections import OrderedDict


import torch.utils.data.dataloader


import collections


import math


import torchvision.transforms as transforms


import random


import torch.nn.functional as F


import torch.utils.data


import torchvision.transforms.functional as tvisf


import pandas


from collections import defaultdict


from scipy.io import loadmat


import torch.utils.model_zoo as model_zoo


from torchvision.models.resnet import model_urls


from torchvision.models.resnet import BasicBlock


from torch import nn


from torch.nn import functional as F


from torch.autograd import Variable


import torch.utils.checkpoint


from copy import deepcopy


from copy import copy


from abc import ABCMeta


from abc import abstractmethod


from torchvision.models.resnet import Bottleneck


import copy


import torch.backends.cudnn


import torch.optim as optim


import time


import pandas as pd


import matplotlib.patches as patches


import matplotlib.pyplot as plt


import matplotlib


import torchvision


import torch.autograd


import functools


import torch.nn


class MultiGPU(nn.DataParallel):
    """Wraps a network to allow simple multi-GPU training."""

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            pass
        return getattr(self.module, item)


class Backbone(nn.Module):
    """Base class for backbone networks. Handles freezing layers etc.
    args:
        frozen_layers  -  Name of layers to freeze. Either list of strings, 'none' or 'all'. Default: 'none'.
    """

    def __init__(self, frozen_layers=()):
        super().__init__()
        if isinstance(frozen_layers, str):
            if frozen_layers.lower() == 'none':
                frozen_layers = ()
            elif frozen_layers.lower() != 'all':
                raise ValueError('Unknown option for frozen layers: "{}". Should be "all", "none" or list of layer names.'.format(frozen_layers))
        self.frozen_layers = frozen_layers
        self._is_frozen_nograd = False

    def train(self, mode=True):
        super().train(mode)
        if mode == True:
            self._set_frozen_to_eval()
        if not self._is_frozen_nograd:
            self._set_frozen_to_nograd()
            self._is_frozen_nograd = True
        return self

    def _set_frozen_to_eval(self):
        if isinstance(self.frozen_layers, str) and self.frozen_layers.lower() == 'all':
            self.eval()
        else:
            for layer in self.frozen_layers:
                getattr(self, layer).eval()

    def _set_frozen_to_nograd(self):
        if isinstance(self.frozen_layers, str) and self.frozen_layers.lower() == 'all':
            for p in self.parameters():
                p.requires_grad_(False)
        else:
            for layer in self.frozen_layers:
                for p in getattr(self, layer).parameters():
                    p.requires_grad_(False)


class h_sigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class h_swish(nn.Module):

    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3.0, self.inplace) / 6.0
        return out * x


class SqueezeBlock(nn.Module):

    def __init__(self, exp_size, divide=4):
        super(SqueezeBlock, self).__init__()
        self.dense = nn.Sequential(nn.Linear(exp_size, exp_size // divide), nn.ReLU(inplace=True), nn.Linear(exp_size // divide, exp_size), h_sigmoid())

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        return out * x


class MobileBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernal_size, stride, nonLinear, SE, exp_size):
        super(MobileBlock, self).__init__()
        self.out_channels = out_channels
        self.nonLinear = nonLinear
        self.SE = SE
        padding = (kernal_size - 1) // 2
        self.use_connect = stride == 1 and in_channels == out_channels
        if self.nonLinear == 'RE':
            activation = nn.ReLU
        else:
            activation = h_swish
        self.conv = nn.Sequential(nn.Conv2d(in_channels, exp_size, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(exp_size), activation(inplace=True))
        self.depth_conv = nn.Sequential(nn.Conv2d(exp_size, exp_size, kernel_size=kernal_size, stride=stride, padding=padding, groups=exp_size), nn.BatchNorm2d(exp_size))
        if self.SE:
            self.squeeze_block = SqueezeBlock(exp_size)
        self.point_conv = nn.Sequential(nn.Conv2d(exp_size, out_channels, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(out_channels), activation(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        out = self.depth_conv(out)
        if self.SE:
            out = self.squeeze_block(out)
        out = self.point_conv(out)
        if self.use_connect:
            return x + out
        else:
            return out


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


class MobileNetV3(nn.Module):

    def __init__(self, model_mode='LARGE', num_classes=1000, multiplier=1.0, dropout_rate=0.0, output_layers=['default']):
        super(MobileNetV3, self).__init__()
        self.num_classes = num_classes
        self.output_layers = output_layers
        if model_mode == 'LARGE':
            layers = [[16, 16, 3, 1, 'RE', False, 16], [16, 24, 3, 2, 'RE', False, 64], [24, 24, 3, 1, 'RE', False, 72], [24, 40, 5, 2, 'RE', True, 72], [40, 40, 5, 1, 'RE', True, 120], [40, 40, 5, 1, 'RE', True, 120], [40, 80, 3, 2, 'HS', False, 240], [80, 80, 3, 1, 'HS', False, 200], [80, 80, 3, 1, 'HS', False, 184], [80, 80, 3, 1, 'HS', False, 184], [80, 112, 3, 1, 'HS', True, 480], [112, 112, 3, 1, 'HS', True, 672], [112, 160, 5, 1, 'HS', True, 672], [160, 160, 5, 2, 'HS', True, 672], [160, 160, 5, 1, 'HS', True, 960]]
            init_conv_out = _make_divisible(16 * multiplier)
            self.init_conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=init_conv_out, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(init_conv_out), h_swish(inplace=True))
            self.layer1 = MobileBlock(16, 16, 3, 1, 'RE', False, 16)
            self.layer2 = nn.Sequential(MobileBlock(16, 24, 3, 2, 'RE', False, 64), MobileBlock(24, 24, 3, 1, 'RE', False, 72))
            self.layer3 = nn.Sequential(MobileBlock(24, 40, 5, 2, 'RE', True, 72), MobileBlock(40, 40, 5, 1, 'RE', True, 120), MobileBlock(40, 40, 5, 1, 'RE', True, 120))
            self.layer4 = nn.Sequential(MobileBlock(40, 80, 3, 2, 'HS', False, 240), MobileBlock(80, 80, 3, 1, 'HS', False, 200), MobileBlock(80, 80, 3, 1, 'HS', False, 184), MobileBlock(80, 80, 3, 1, 'HS', False, 184))
            self.layer5 = nn.Sequential(MobileBlock(80, 112, 3, 1, 'HS', True, 480), MobileBlock(112, 112, 3, 1, 'HS', True, 672))
            self.layer6 = nn.Sequential(MobileBlock(112, 160, 5, 1, 'HS', True, 672), MobileBlock(160, 160, 5, 2, 'HS', True, 672), MobileBlock(160, 160, 5, 1, 'HS', True, 960))
            out_conv1_in = _make_divisible(160 * multiplier)
            out_conv1_out = _make_divisible(960 * multiplier)
            self.out_conv1 = nn.Sequential(nn.Conv2d(out_conv1_in, out_conv1_out, kernel_size=1, stride=1), nn.BatchNorm2d(out_conv1_out), h_swish(inplace=True))
            out_conv2_in = _make_divisible(960 * multiplier)
            out_conv2_out = _make_divisible(1280 * multiplier)
            self.out_conv2 = nn.Sequential(nn.Conv2d(out_conv2_in, out_conv2_out, kernel_size=1, stride=1), h_swish(inplace=True), nn.Dropout(dropout_rate), nn.Conv2d(out_conv2_out, self.num_classes, kernel_size=1, stride=1))
        elif model_mode == 'SMALL':
            layers = [[16, 16, 3, 2, 'RE', True, 16], [16, 24, 3, 2, 'RE', False, 72], [24, 24, 3, 1, 'RE', False, 88], [24, 40, 5, 2, 'RE', True, 96], [40, 40, 5, 1, 'RE', True, 240], [40, 40, 5, 1, 'RE', True, 240], [40, 48, 5, 1, 'HS', True, 120], [48, 48, 5, 1, 'HS', True, 144], [48, 96, 5, 2, 'HS', True, 288], [96, 96, 5, 1, 'HS', True, 576], [96, 96, 5, 1, 'HS', True, 576]]
            init_conv_out = _make_divisible(16 * multiplier)
            self.init_conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=init_conv_out, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(init_conv_out), h_swish(inplace=True))
            self.block = []
            for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size in layers:
                in_channels = _make_divisible(in_channels * multiplier)
                out_channels = _make_divisible(out_channels * multiplier)
                exp_size = _make_divisible(exp_size * multiplier)
                self.block.append(MobileBlock(in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size))
            out_conv1_in = _make_divisible(96 * multiplier)
            out_conv1_out = _make_divisible(576 * multiplier)
            self.out_conv1 = nn.Sequential(nn.Conv2d(out_conv1_in, out_conv1_out, kernel_size=1, stride=1), SqueezeBlock(out_conv1_out), nn.BatchNorm2d(out_conv1_out), h_swish(inplace=True))
            out_conv2_in = _make_divisible(576 * multiplier)
            out_conv2_out = _make_divisible(1280 * multiplier)
            self.out_conv2 = nn.Sequential(nn.Conv2d(out_conv2_in, out_conv2_out, kernel_size=1, stride=1), h_swish(inplace=True), nn.Dropout(dropout_rate), nn.Conv2d(out_conv2_out, self.num_classes, kernel_size=1, stride=1))
        self.apply(_weights_init)

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, output_layers=None):
        outputs = OrderedDict()
        if output_layers is None:
            output_layers = self.output_layers
        out = self.init_conv(x)
        if self._add_output_and_check('init_conv', out, outputs, output_layers):
            return outputs
        out = self.layer1(out)
        if self._add_output_and_check('layer1', out, outputs, output_layers):
            return outputs
        out = self.layer2(out)
        if self._add_output_and_check('layer2', out, outputs, output_layers):
            return outputs
        out = self.layer3(out)
        if self._add_output_and_check('layer3', out, outputs, output_layers):
            return outputs
        out = self.layer4(out)
        if self._add_output_and_check('layer4', out, outputs, output_layers):
            return outputs
        out = self.layer5(out)
        if self._add_output_and_check('layer5', out, outputs, output_layers):
            return outputs
        out = self.layer6(out)
        if self._add_output_and_check('layer6', out, outputs, output_layers):
            return outputs
        out = self.out_conv1(out)
        if self._add_output_and_check('layer_out', out, outputs, output_layers):
            return outputs
        batch, channels, height, width = out.size()
        out = F.avg_pool2d(out, kernel_size=[height, width])
        out = self.out_conv2(out).view(batch, -1)
        if len(output_layers) == 1 and output_layers[0] == 'default':
            return out
        return outputs


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        if use_bn:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride_1x1=1, stride_3x3=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride_1x1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride_3x3, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

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


class ResNet(Backbone):
    """ ResNet network module. Allows extracting specific feature blocks."""

    def __init__(self, block, layers, output_layers, num_classes=1000, inplanes=64, dilation_factor=1, frozen_layers=()):
        self.inplanes = inplanes
        super(ResNet, self).__init__(frozen_layers=frozen_layers)
        self.output_layers = output_layers
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stride = [(int(l > 1) + 1) for l in range(1, 4)]
        self.layer1 = self._make_layer(block, inplanes, layers[0], stride=stride[0], dilation=max(dilation_factor // 8, 1))
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=stride[1], dilation=max(dilation_factor // 4, 1))
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=stride[2], dilation=max(dilation_factor // 2, 1))
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=stride[2], dilation=dilation_factor)
        out_feature_strides = {'conv1': 4, 'layer1': 4, 'layer2': 4 * stride[0], 'layer3': 4 * stride[0] * stride[1], 'layer4': 4 * stride[0] * stride[1] * stride[2]}
        if isinstance(self.layer1[0], Bottleneck):
            base_num_channels = 4 * inplanes
            out_feature_channels = {'conv1': inplanes, 'layer1': base_num_channels, 'layer2': base_num_channels * 2, 'layer3': base_num_channels * 4, 'layer4': base_num_channels * 8}
        else:
            raise Exception('block not supported')
        self._out_feature_strides = out_feature_strides
        self._out_feature_channels = out_feature_channels
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def out_feature_strides(self, layer=None):
        if layer is None:
            return self._out_feature_strides
        else:
            return self._out_feature_strides[layer]

    def out_feature_channels(self, layer=None):
        if layer is None:
            return self._out_feature_channels
        else:
            return self._out_feature_channels[layer]

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, stride_in_1x1=True):
        downsample = None
        if self.inplanes != planes * block.expansion:
            down_stride = stride if dilation == 1 else 1
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=down_stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        if dilation > 1:
            stride = 1
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride_1x1, stride_3x3, downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, output_layers=None):
        """ Forward pass with input x. The output_layers specify the feature blocks which must be returned """
        outputs = OrderedDict()
        if output_layers is None:
            output_layers = self.output_layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self._add_output_and_check('conv1', x, outputs, output_layers):
            return outputs
        x = self.layer1(x)
        if self._add_output_and_check('layer1', x, outputs, output_layers):
            return outputs
        x = self.layer2(x)
        if self._add_output_and_check('layer2', x, outputs, output_layers):
            return outputs
        x = self.layer3(x)
        if self._add_output_and_check('layer3', x, outputs, output_layers):
            return outputs
        x = self.layer4(x)
        if self._add_output_and_check('layer4', x, outputs, output_layers):
            return outputs
        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x
        raise ValueError('output_layer is wrong.')


class SpatialCrossMapLRN(nn.Module):

    def __init__(self, local_size=1, alpha=1.0, beta=0.75, k=1, ACROSS_CHANNELS=True):
        super(SpatialCrossMapLRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1), stride=1, padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size, stride=1, padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x


class ResNetVGGm1(Backbone):

    def __init__(self, block, layers, output_layers, num_classes=1000, frozen_layers=()):
        self.inplanes = 64
        super(ResNetVGGm1, self).__init__(frozen_layers=frozen_layers)
        self.output_layers = output_layers
        self.vggmconv1 = nn.Conv2d(3, 96, (7, 7), (2, 2), padding=3)
        self.vgglrn = SpatialCrossMapLRN(5, 0.0005, 0.75, 2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, output_layers=None):
        outputs = OrderedDict()
        if output_layers is None:
            output_layers = self.output_layers
        if 'vggconv1' in output_layers:
            c1 = self.vgglrn(self.relu(self.vggmconv1(x)))
            if self._add_output_and_check('vggconv1', c1, outputs, output_layers):
                return outputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self._add_output_and_check('conv1', x, outputs, output_layers):
            return outputs
        x = self.maxpool(x)
        x = self.layer1(x)
        if self._add_output_and_check('layer1', x, outputs, output_layers):
            return outputs
        x = self.layer2(x)
        if self._add_output_and_check('layer2', x, outputs, output_layers):
            return outputs
        x = self.layer3(x)
        if self._add_output_and_check('layer3', x, outputs, output_layers):
            return outputs
        x = self.layer4(x)
        if self._add_output_and_check('layer4', x, outputs, output_layers):
            return outputs
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self._add_output_and_check('fc', x, outputs, output_layers):
            return outputs
        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x
        raise ValueError('output_layer is wrong.')


class ATOMnet(nn.Module):
    """ ATOM network module"""

    def __init__(self, feature_extractor, bb_regressor, bb_regressor_layer, extractor_grad=True):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(ATOMnet, self).__init__()
        self.feature_extractor = feature_extractor
        self.bb_regressor = bb_regressor
        self.bb_regressor_layer = bb_regressor_layer
        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs, test_imgs, train_bb, test_proposals):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        num_sequences = train_imgs.shape[-4]
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1
        num_test_images = test_imgs.shape[0] if test_imgs.dim() == 5 else 1
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))
        train_feat_iou = [feat for feat in train_feat.values()]
        test_feat_iou = [feat for feat in test_feat.values()]
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou, train_bb.reshape(num_train_images, num_sequences, 4), test_proposals.reshape(num_train_images, num_sequences, -1, 4))
        return iou_pred

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)


class LinearBlock(nn.Module):

    def __init__(self, in_planes, out_planes, input_sz, bias=True, batch_norm=True, relu=True):
        super().__init__()
        self.linear = nn.Linear(in_planes * input_sz * input_sz, out_planes, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if batch_norm else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.linear(x.reshape(x.shape[0], -1))
        if self.bn is not None:
            x = self.bn(x.reshape(x.shape[0], x.shape[1], 1, 1))
        if self.relu is not None:
            x = self.relu(x)
        return x.reshape(x.shape[0], -1)


def conv(ic, oc, ksize, bias=True, dilation=1, stride=1):
    return nn.Conv2d(ic, oc, ksize, padding=ksize // 2, bias=bias, dilation=dilation, stride=stride)


class AtomIoUNet(nn.Module):
    """Network module for IoU prediction. Refer to the ATOM paper for an illustration of the architecture.
    It uses two backbone feature layers as input.
    args:
        input_dim:  Feature dimensionality of the two input backbone layers.
        pred_input_dim:  Dimensionality input the the prediction network.
        pred_inter_dim:  Intermediate dimensionality in the prediction network."""

    def __init__(self, input_dim=(128, 256), pred_input_dim=(256, 256), pred_inter_dim=(256, 256)):
        super().__init__()
        self.conv3_1r = conv(input_dim[0], 128, kernel_size=3, stride=1)
        self.conv3_1t = conv(input_dim[0], 256, kernel_size=3, stride=1)
        self.conv3_2t = conv(256, pred_input_dim[0], kernel_size=3, stride=1)
        self.prroi_pool3r = PrRoIPool2D(3, 3, 1 / 8)
        self.prroi_pool3t = PrRoIPool2D(5, 5, 1 / 8)
        self.fc3_1r = conv(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv4_1r = conv(input_dim[1], 256, kernel_size=3, stride=1)
        self.conv4_1t = conv(input_dim[1], 256, kernel_size=3, stride=1)
        self.conv4_2t = conv(256, pred_input_dim[1], kernel_size=3, stride=1)
        self.prroi_pool4r = PrRoIPool2D(1, 1, 1 / 16)
        self.prroi_pool4t = PrRoIPool2D(3, 3, 1 / 16)
        self.fc34_3r = conv(256 + 256, pred_input_dim[0], kernel_size=1, stride=1, padding=0)
        self.fc34_4r = conv(256 + 256, pred_input_dim[1], kernel_size=1, stride=1, padding=0)
        self.fc3_rt = LinearBlock(pred_input_dim[0], pred_inter_dim[0], 5)
        self.fc4_rt = LinearBlock(pred_input_dim[1], pred_inter_dim[1], 3)
        self.iou_predictor = nn.Linear(pred_inter_dim[0] + pred_inter_dim[1], 1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def forward(self, feat1, feat2, bb1, proposals2):
        """Runs the ATOM IoUNet during training operation.
        This forward pass is mainly used for training. Call the individual functions during tracking instead.
        args:
            feat1:  Features from the reference frames (4 or 5 dims).
            feat2:  Features from the test frames (4 or 5 dims).
            bb1:  Target boxes (x,y,w,h) in image coords in the reference samples. Dims (images, sequences, 4).
            proposals2:  Proposal boxes for which the IoU will be predicted (images, sequences, num_proposals, 4)."""
        assert bb1.dim() == 3
        assert proposals2.dim() == 4
        num_images = proposals2.shape[0]
        num_sequences = proposals2.shape[1]
        feat1 = [(f[0, ...] if f.dim() == 5 else f.reshape(-1, num_sequences, *f.shape[-3:])[0, ...]) for f in feat1]
        bb1 = bb1[0, ...]
        modulation = self.get_modulation(feat1, bb1)
        iou_feat = self.get_iou_feat(feat2)
        modulation = [f.reshape(1, num_sequences, -1).repeat(num_images, 1, 1).reshape(num_sequences * num_images, -1) for f in modulation]
        proposals2 = proposals2.reshape(num_sequences * num_images, -1, 4)
        pred_iou = self.predict_iou(modulation, iou_feat, proposals2)
        return pred_iou.reshape(num_images, num_sequences, -1)

    def predict_iou(self, modulation, feat, proposals):
        """Predicts IoU for the give proposals.
        args:
            modulation:  Modulation vectors for the targets. Dims (batch, feature_dim).
            feat:  IoU features (from get_iou_feat) for test images. Dims (batch, feature_dim, H, W).
            proposals:  Proposal boxes for which the IoU will be predicted (batch, num_proposals, 4)."""
        fc34_3_r, fc34_4_r = modulation
        c3_t, c4_t = feat
        batch_size = c3_t.size()[0]
        c3_t_att = c3_t * fc34_3_r.reshape(batch_size, -1, 1, 1)
        c4_t_att = c4_t * fc34_4_r.reshape(batch_size, -1, 1, 1)
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1)
        num_proposals_per_batch = proposals.shape[1]
        proposals_xyxy = torch.cat((proposals[:, :, 0:2], proposals[:, :, 0:2] + proposals[:, :, 2:4]), dim=2)
        roi2 = torch.cat((batch_index.reshape(batch_size, -1, 1).expand(-1, num_proposals_per_batch, -1), proposals_xyxy), dim=2)
        roi2 = roi2.reshape(-1, 5)
        roi3t = self.prroi_pool3t(c3_t_att, roi2)
        roi4t = self.prroi_pool4t(c4_t_att, roi2)
        fc3_rt = self.fc3_rt(roi3t)
        fc4_rt = self.fc4_rt(roi4t)
        fc34_rt_cat = torch.cat((fc3_rt, fc4_rt), dim=1)
        iou_pred = self.iou_predictor(fc34_rt_cat).reshape(batch_size, num_proposals_per_batch)
        return iou_pred

    def get_modulation(self, feat, bb):
        """Get modulation vectors for the targets.
        args:
            feat: Backbone features from reference images. Dims (batch, feature_dim, H, W).
            bb:  Target boxes (x,y,w,h) in image coords in the reference samples. Dims (batch, 4)."""
        feat3_r, feat4_r = feat
        c3_r = self.conv3_1r(feat3_r)
        batch_size = bb.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1)
        bb = bb.clone()
        bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
        roi1 = torch.cat((batch_index, bb), dim=1)
        roi3r = self.prroi_pool3r(c3_r, roi1)
        c4_r = self.conv4_1r(feat4_r)
        roi4r = self.prroi_pool4r(c4_r, roi1)
        fc3_r = self.fc3_1r(roi3r)
        fc34_r = torch.cat((fc3_r, roi4r), dim=1)
        fc34_3_r = self.fc34_3r(fc34_r)
        fc34_4_r = self.fc34_4r(fc34_r)
        return fc34_3_r, fc34_4_r

    def get_iou_feat(self, feat2):
        """Get IoU prediction features from a 4 or 5 dimensional backbone input."""
        feat2 = [(f.reshape(-1, *f.shape[-3:]) if f.dim() == 5 else f) for f in feat2]
        feat3_t, feat4_t = feat2
        c3_t = self.conv3_2t(self.conv3_1t(feat3_t))
        c4_t = self.conv4_2t(self.conv4_1t(feat4_t))
        return c3_t, c4_t


def conv_block(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, batch_norm=True, relu=True, padding_mode='zeros'):
    layers = []
    assert padding_mode == 'zeros' or padding_mode == 'replicate'
    if padding_mode == 'replicate' and padding > 0:
        assert isinstance(padding, int)
        layers.append(nn.ReflectionPad2d(padding))
        padding = 0
    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_planes))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ConvGRUCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, padding_mode='zeros'):
        """ Referenced from https://github.com/happyjin/ConvGRU-pytorch"""
        super(ConvGRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        if padding_mode == 'zeros':
            if not isinstance(kernel_size, (list, tuple)):
                kernel_size = kernel_size, kernel_size
            padding = kernel_size[0] // 2, kernel_size[1] // 2
            self.conv_reset = nn.Conv2d(input_dim + hidden_dim, self.hidden_dim, kernel_size, padding=padding)
            self.conv_update = nn.Conv2d(input_dim + hidden_dim, self.hidden_dim, kernel_size, padding=padding)
            self.conv_state_new = nn.Conv2d(input_dim + hidden_dim, self.hidden_dim, kernel_size, padding=padding)
        else:
            self.conv_reset = conv_block(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2), batch_norm=False, relu=False, padding_mode=padding_mode)
            self.conv_update = conv_block(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2), batch_norm=False, relu=False, padding_mode=padding_mode)
            self.conv_state_new = conv_block(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2), batch_norm=False, relu=False, padding_mode=padding_mode)

    def forward(self, input, state_cur):
        input_state_cur = torch.cat([input, state_cur], dim=1)
        reset_gate = torch.sigmoid(self.conv_reset(input_state_cur))
        update_gate = torch.sigmoid(self.conv_update(input_state_cur))
        input_state_cur_reset = torch.cat([input, reset_gate * state_cur], dim=1)
        state_new = torch.tanh(self.conv_state_new(input_state_cur_reset))
        state_next = (1.0 - update_gate) * state_cur + update_gate * state_new
        return state_next


def remap_cost_volume(cost_volume):
    """

    :param cost_volume: cost volume of shape (batch, (2*md-1)*(2*md-1), rows, cols), where md is the maximum displacement
                        allowed when computing the cost volume.
    :return: cost_volume_remapped: The input cost volume is remapped to shape (batch, rows, cols, rows, cols)
    """
    if cost_volume.dim() != 4:
        raise ValueError('input cost_volume should have 4 dimensions')
    [batch_size, d_, num_rows, num_cols] = cost_volume.size()
    d_sqrt_ = np.sqrt(d_)
    if not d_sqrt_.is_integer():
        raise ValueError('Invalid cost volume')
    cost_volume = cost_volume.view(batch_size, int(d_sqrt_), int(d_sqrt_), num_rows, num_cols)
    cost_volume_remapped = torch.zeros((batch_size, num_rows, num_cols, num_rows, num_cols), dtype=cost_volume.dtype, device=cost_volume.device)
    if cost_volume.size()[1] % 2 != 1:
        raise ValueError
    md = int((cost_volume.size()[1] - 1) / 2)
    for r in range(num_rows):
        for c in range(num_cols):
            r1_ = r - md
            r2_ = r1_ + 2 * md + 1
            c1_ = c - md
            c2_ = c1_ + 2 * md + 1
            r1_pad_ = max(-r1_, 0)
            r2_pad_ = max(r2_ - cost_volume_remapped.shape[1], 0)
            c1_pad_ = max(-c1_, 0)
            c2_pad_ = max(c2_ - cost_volume_remapped.shape[2], 0)
            d_ = cost_volume.size()[1]
            cost_volume_remapped[:, r1_ + r1_pad_:r2_ - r2_pad_, c1_ + c1_pad_:c2_ - c2_pad_, r, c] = cost_volume[:, r1_pad_:d_ - r2_pad_, c1_pad_:d_ - c2_pad_, r, c]
    return cost_volume_remapped


class CostVolume(nn.Module):

    def __init__(self, kernel_size, max_displacement, stride=1, abs_coordinate_output=False):
        super().__init__()
        self.correlation_layer = SpatialCorrelationSampler(kernel_size, 2 * max_displacement + 1, stride, int((kernel_size - 1) / 2))
        self.abs_coordinate_output = abs_coordinate_output

    def forward(self, feat1, feat2):
        assert feat1.dim() == 4 and feat2.dim() == 4, 'Expect 4 dimensional inputs'
        batch_size = feat1.shape[0]
        cost_volume = self.correlation_layer(feat1, feat2)
        if self.abs_coordinate_output:
            cost_volume = cost_volume.view(batch_size, -1, cost_volume.shape[-2], cost_volume.shape[-1])
            cost_volume = remap_cost_volume(cost_volume)
        return cost_volume.view(batch_size, -1, cost_volume.shape[-2], cost_volume.shape[-1])


def shift_features(feat, relative_translation_vector):
    T_mat = torch.eye(2).repeat(feat.shape[0], 1, 1)
    T_mat = torch.cat((T_mat, relative_translation_vector.view(-1, 2, 1)), dim=2)
    grid = F.affine_grid(T_mat, feat.shape)
    feat_out = F.grid_sample(feat, grid)
    return feat_out


class PredictorWrapper(nn.Module):

    def __init__(self, cost_volume, predictor):
        super().__init__()
        self.cost_volume = cost_volume
        self.predictor = predictor
        self.fix_coordinate_shift = True

    def forward(self, data):
        input1 = data['input1']
        input2 = data['input2']
        label_prev = data.get('label_prev', None)
        dimp_score_cur = data['dimp_score_cur']
        state_prev = data['state_prev']
        score_shape = dimp_score_cur.shape
        if isinstance(input1, (tuple, list)):
            feat1 = [self.extract_motion_feat(in1) for in1 in input1]
            feat1 = [f1.view(-1, *f1.shape[-3:]) for f1 in feat1]
        else:
            feat1 = self.extract_motion_feat(input1)
            feat1 = feat1.view(-1, *feat1.shape[-3:])
        feat2 = self.extract_motion_feat(input2)
        feat2 = feat2.view(-1, *feat2.shape[-3:])
        dimp_score_cur = dimp_score_cur.view(-1, 1, *dimp_score_cur.shape[-2:])
        if isinstance(input1, (tuple, list)):
            cost_volume = [self.compute_cost_volume(f1, feat2, True) for f1 in feat1]
        else:
            cost_volume = self.compute_cost_volume(feat1, feat2, True)
        feat_map_size = torch.tensor([dimp_score_cur.shape[-1], dimp_score_cur.shape[-2]]).view(1, 2).float()
        if self.fix_coordinate_shift:
            shift_value = -torch.ones(dimp_score_cur.shape[0], 2) * 0.5 / feat_map_size
            if label_prev is not None:
                label_prev_shape = label_prev.shape
                label_prev = shift_features(label_prev.clone().view(-1, 1, *dimp_score_cur.shape[-2:]), shift_value).view(label_prev_shape)
            dimp_score_cur = shift_features(dimp_score_cur.clone(), shift_value)
        pred_response, state_new, auxiliary_outputs = self.predictor(cost_volume, state_prev, dimp_score_cur, label_prev)
        pred_response = pred_response.view(score_shape)
        if self.fix_coordinate_shift:
            shift_value = torch.ones(dimp_score_cur.shape[0], 2) * 0.5 / feat_map_size
            if 'is_target' in auxiliary_outputs:
                auxiliary_outputs['is_target'] = shift_features(auxiliary_outputs['is_target'].view(-1, 1, *dimp_score_cur.shape[-2:]), shift_value).view(score_shape)
            if 'is_target_after_prop' in auxiliary_outputs:
                auxiliary_outputs['is_target_after_prop'] = shift_features(auxiliary_outputs['is_target_after_prop'].view(-1, 1, *dimp_score_cur.shape[-2:]), shift_value).view(score_shape)
            if 'is_target_new' in auxiliary_outputs:
                auxiliary_outputs['is_target_new'] = shift_features(auxiliary_outputs['is_target_new'].view(-1, 1, *dimp_score_cur.shape[-2:]), shift_value).view(score_shape)
            pred_response = shift_features(pred_response.view(-1, 1, *dimp_score_cur.shape[-2:]), shift_value).view(score_shape)
        output = {'response': pred_response, 'state_cur': state_new, 'auxiliary_outputs': auxiliary_outputs}
        return output

    def compute_cost_volume(self, feat_prev, feat_cur, use_current_frame_as_ref):
        if use_current_frame_as_ref:
            cost_volume = self.cost_volume(feat_cur, feat_prev)
        else:
            cost_volume = self.cost_volume(feat_prev, feat_cur)
        return cost_volume

    def extract_motion_feat(self, backbone_feat):
        backbone_feat = backbone_feat.view(-1, backbone_feat.shape[-3], backbone_feat.shape[-2], backbone_feat.shape[-1])
        return backbone_feat

    def predict_response(self, data, dimp_thresh=None, output_window=None):
        feat1 = data['feat1']
        feat2 = data['feat2']
        label_prev = data.get('label_prev', None)
        dimp_score_cur = data['dimp_score_cur']
        state_prev = data['state_prev']
        score_shape = dimp_score_cur.shape
        if isinstance(feat1, (tuple, list)):
            feat1 = [f1.view(-1, *f1.shape[-3:]) for f1 in feat1]
        else:
            feat1 = feat1.view(-1, *feat1.shape[-3:])
        feat2 = feat2.view(-1, *feat2.shape[-3:])
        dimp_score_cur = dimp_score_cur.view(-1, 1, *dimp_score_cur.shape[-2:])
        if isinstance(feat1, (tuple, list)):
            cost_volume = [self.compute_cost_volume(f1, feat2, True) for f1 in feat1]
        else:
            cost_volume = self.compute_cost_volume(feat1, feat2, True)
        feat_map_size = torch.tensor([dimp_score_cur.shape[-1], dimp_score_cur.shape[-2]]).view(1, 2).float()
        if self.fix_coordinate_shift:
            shift_value = -torch.ones(dimp_score_cur.shape[0], 2) * 0.5 / feat_map_size
            if label_prev is not None:
                label_prev_shape = label_prev.shape
                label_prev = shift_features(label_prev.clone().view(-1, 1, *dimp_score_cur.shape[-2:]), shift_value).view(label_prev_shape)
            dimp_score_cur = shift_features(dimp_score_cur.clone(), shift_value)
        pred_response, state_new, auxiliary_outputs = self.predictor(cost_volume, state_prev, dimp_score_cur, label_prev, dimp_thresh, output_window)
        pred_response = pred_response.view(score_shape)
        if self.fix_coordinate_shift:
            shift_value = torch.ones(dimp_score_cur.shape[0], 2) * 0.5 / feat_map_size
            if 'is_target' in auxiliary_outputs:
                auxiliary_outputs['is_target'] = shift_features(auxiliary_outputs['is_target'].view(-1, 1, *dimp_score_cur.shape[-2:]), shift_value).view(score_shape)
            if 'is_target_after_prop' in auxiliary_outputs:
                auxiliary_outputs['is_target_after_prop'] = shift_features(auxiliary_outputs['is_target_after_prop'].view(-1, 1, *dimp_score_cur.shape[-2:]), shift_value).view(score_shape)
            if 'is_target_new' in auxiliary_outputs:
                auxiliary_outputs['is_target_new'] = shift_features(auxiliary_outputs['is_target_new'].view(-1, 1, *dimp_score_cur.shape[-2:]), shift_value).view(score_shape)
            pred_response = shift_features(pred_response.view(-1, 1, *dimp_score_cur.shape[-2:]), shift_value).view(score_shape)
        output = {'response': pred_response, 'state_cur': state_new, 'auxiliary_outputs': auxiliary_outputs, 'cost_volume': cost_volume}
        return output


class ResponsePredictor(nn.Module):

    def __init__(self, state_dim=8, representation_predictor_dims=(64, 32), gru_ksz=3, prev_max_pool_ksz=1, conf_measure='max', dimp_thresh=None):
        super().__init__()
        self.prev_max_pool_ksz = prev_max_pool_ksz
        self.conf_measure = conf_measure
        self.dimp_thresh = dimp_thresh
        cvproc_ksz = [3, 3]
        use_bn = True
        padding_val = [int((s - 1) / 2) for s in cvproc_ksz]
        self.cost_volume_proc1 = nn.Sequential(conv_block(1, 8, kernel_size=cvproc_ksz[0], stride=1, padding=padding_val[0], batch_norm=use_bn, relu=True), conv_block(8, 1, kernel_size=cvproc_ksz[1], stride=1, padding=padding_val[1], batch_norm=use_bn, relu=False))
        self.cost_volume_proc2 = nn.Sequential(conv_block(1, 8, kernel_size=cvproc_ksz[0], stride=1, padding=padding_val[0], batch_norm=use_bn, relu=True), conv_block(8, 1, kernel_size=cvproc_ksz[1], stride=1, padding=padding_val[1], batch_norm=use_bn, relu=False))
        in_dim = state_dim + 1 + (conf_measure != 'none')
        representation_predictor_list = []
        for out_dim in representation_predictor_dims:
            representation_predictor_list.append(conv_block(in_dim, out_dim, kernel_size=3, stride=1, padding=1, batch_norm=False, relu=True))
            in_dim = out_dim
        self.representation_predictor = nn.Sequential(*representation_predictor_list)
        self.representation_dim = in_dim
        self.response_predictor = nn.Sequential(conv_block(in_dim, 1, kernel_size=3, stride=1, padding=1, batch_norm=False, relu=False), nn.Sigmoid())
        self.state_predictor = ConvGRUCell(4, state_dim, gru_ksz)
        self.init_hidden_state_predictor = nn.Sequential(conv_block(1, state_dim, kernel_size=3, stride=1, padding=1, batch_norm=False, relu=False, bias=False), nn.Tanh())
        self.is_target_predictor = nn.Sequential(conv_block(state_dim, 4, kernel_size=gru_ksz, stride=1, padding=int(gru_ksz // 2), batch_norm=False, relu=True), conv_block(4, 1, kernel_size=gru_ksz, stride=1, padding=int(gru_ksz // 2), batch_norm=False, relu=False))
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, cost_volume, state_prev, dimp_score_cur, init_label=None, dimp_thresh=None, output_window=None):
        if dimp_thresh is None:
            dimp_thresh = self.dimp_thresh
        auxiliary_outputs = {}
        num_sequences = cost_volume.shape[0]
        feat_sz = cost_volume.shape[-2:]
        cost_volume = cost_volume.view(-1, 1, feat_sz[0], feat_sz[1])
        cost_volume_p1 = self.cost_volume_proc1(cost_volume).view(-1, feat_sz[0] * feat_sz[1])
        cost_volume_p1 = F.softmax(cost_volume_p1, dim=1)
        cost_volume_p2 = self.cost_volume_proc2(cost_volume_p1.view(-1, 1, feat_sz[0], feat_sz[1]))
        cost_volume_p2 = cost_volume_p2.view(num_sequences, -1, feat_sz[0], feat_sz[1])
        cost_volume_p2 = F.softmax(cost_volume_p2, dim=1)
        cost_volume_p2 = cost_volume_p2.view(num_sequences, -1, 1, feat_sz[0], feat_sz[1])
        auxiliary_outputs['cost_volume_processed'] = cost_volume_p2
        if state_prev is None:
            init_hidden_state = self.init_hidden_state_predictor(init_label.view(num_sequences, 1, feat_sz[0], feat_sz[1]))
            state_prev_ndhw = init_hidden_state
        else:
            state_prev_ndhw = state_prev
        is_target = self.is_target_predictor(state_prev_ndhw)
        auxiliary_outputs['is_target'] = is_target
        state_prev_ndhw = state_prev_ndhw.view(num_sequences, -1, feat_sz[0], feat_sz[1])
        state_prev_nhwd = state_prev_ndhw.permute(0, 2, 3, 1).contiguous().view(num_sequences, feat_sz[0] * feat_sz[1], -1, 1, 1).expand(-1, -1, -1, feat_sz[0], feat_sz[1])
        propagation_weight_norm = cost_volume_p2.view(num_sequences, feat_sz[0] * feat_sz[1], 1, feat_sz[0], feat_sz[1])
        if self.prev_max_pool_ksz > 1:
            raise NotImplementedError
        if self.conf_measure == 'max':
            propagation_conf = propagation_weight_norm.view(num_sequences, -1, feat_sz[0], feat_sz[1]).max(dim=1)[0]
        elif self.conf_measure == 'entropy':
            propagation_conf = propagation_weight_norm.view(num_sequences, -1, feat_sz[0], feat_sz[1])
            propagation_conf = -(propagation_conf * (propagation_conf + 0.0001).log()).sum(dim=1)
        auxiliary_outputs['propagation_weights'] = propagation_weight_norm
        propagated_h = (propagation_weight_norm * state_prev_nhwd).sum(dim=1)
        propagated_h = propagated_h.view(num_sequences, -1, feat_sz[0], feat_sz[1])
        auxiliary_outputs['propagated_h'] = propagated_h.clone()
        is_target_after_prop = self.is_target_predictor(propagated_h)
        auxiliary_outputs['is_target_after_prop'] = is_target_after_prop
        if self.conf_measure != 'none':
            propagation_conf = propagation_conf.view(num_sequences, 1, feat_sz[0], feat_sz[1])
            auxiliary_outputs['propagation_conf'] = propagation_conf
            predictor_input = torch.cat((propagated_h, dimp_score_cur.view(num_sequences, 1, *dimp_score_cur.shape[-2:]), propagation_conf), dim=1)
        else:
            predictor_input = torch.cat((propagated_h, dimp_score_cur.view(num_sequences, 1, *dimp_score_cur.shape[-2:])), dim=1)
        resp_representation = self.representation_predictor(predictor_input)
        fused_prediction = self.response_predictor(resp_representation)
        auxiliary_outputs['fused_score_orig'] = fused_prediction.clone()
        if dimp_thresh is not None:
            fused_prediction = fused_prediction * (dimp_score_cur > dimp_thresh).float()
        if output_window is not None:
            fused_prediction = fused_prediction * output_window
        scores_cat = torch.cat((dimp_score_cur, fused_prediction), dim=1)
        scores_cat_pool = F.adaptive_max_pool2d(scores_cat, 1).view(scores_cat.shape[0], scores_cat.shape[1], 1, 1).expand(-1, -1, feat_sz[0], feat_sz[1])
        state_gru_input = torch.cat((scores_cat, scores_cat_pool), dim=1)
        state_new = self.state_predictor(state_gru_input, propagated_h)
        is_target_new = self.is_target_predictor(state_new)
        auxiliary_outputs['is_target_new'] = is_target_new
        return fused_prediction, state_new, auxiliary_outputs


class CenterShiftFeatures(nn.Module):

    def __init__(self, feature_stride):
        super().__init__()
        self.feature_stride = feature_stride

    def forward(self, feat, anno):
        anno = anno.view(-1, 4)
        c_x = (anno[:, 0] + anno[:, 2] * 0.5) / self.feature_stride
        c_y = (anno[:, 1] + anno[:, 3] * 0.5) / self.feature_stride
        t_x = 2 * (c_x - feat.shape[-1] * 0.5) / feat.shape[-1]
        t_y = 2 * (c_y - feat.shape[-2] * 0.5) / feat.shape[-2]
        t = torch.cat((t_x.view(-1, 1), t_y.view(-1, 1)), dim=1)
        feat_out = shift_features(feat, t)
        return feat_out


class MLU(nn.Module):
    """MLU activation
    """

    def __init__(self, min_val, inplace=False):
        super().__init__()
        self.min_val = min_val
        self.inplace = inplace

    def forward(self, input):
        return F.elu(F.leaky_relu(input, 1 / self.min_val, inplace=self.inplace), self.min_val, inplace=self.inplace)


class LeakyReluPar(nn.Module):
    """LeakyRelu parametric activation
    """

    def forward(self, x, a):
        return (1.0 - a) / 2.0 * torch.abs(x) + (1.0 + a) / 2.0 * x


class LeakyReluParDeriv(nn.Module):
    """Derivative of the LeakyRelu parametric activation, wrt x.
    """

    def forward(self, x, a):
        return (1.0 - a) / 2.0 * torch.sign(x.detach()) + (1.0 + a) / 2.0


class BentIdentPar(nn.Module):
    """BentIdent parametric activation
    """

    def __init__(self, b=1.0):
        super().__init__()
        self.b = b

    def forward(self, x, a):
        return (1.0 - a) / 2.0 * (torch.sqrt(x * x + 4.0 * self.b * self.b) - 2.0 * self.b) + (1.0 + a) / 2.0 * x


class BentIdentParDeriv(nn.Module):
    """BentIdent parametric activation deriv
    """

    def __init__(self, b=1.0):
        super().__init__()
        self.b = b

    def forward(self, x, a):
        return (1.0 - a) / 2.0 * (x / torch.sqrt(x * x + 4.0 * self.b * self.b)) + (1.0 + a) / 2.0


class DistanceMap(nn.Module):
    """Generate a distance map from a origin center location.
    args:
        num_bins:  Number of bins in the map.
        bin_displacement:  Displacement of the bins.
    """

    def __init__(self, num_bins, bin_displacement=1.0):
        super().__init__()
        self.num_bins = num_bins
        self.bin_displacement = bin_displacement

    def forward(self, center, output_sz):
        """Create the distance map.
        args:
            center: Torch tensor with (y,x) center position. Dims (batch, 2)
            output_sz: Size of output distance map. 2-dimensional tuple."""
        center = center.view(-1, 2)
        bin_centers = torch.arange(self.num_bins, dtype=torch.float32, device=center.device).view(1, -1, 1, 1)
        k0 = torch.arange(output_sz[0], dtype=torch.float32, device=center.device).view(1, 1, -1, 1)
        k1 = torch.arange(output_sz[1], dtype=torch.float32, device=center.device).view(1, 1, 1, -1)
        d0 = k0 - center[:, 0].view(-1, 1, 1, 1)
        d1 = k1 - center[:, 1].view(-1, 1, 1, 1)
        dist = torch.sqrt(d0 * d0 + d1 * d1)
        bin_diff = dist / self.bin_displacement - bin_centers
        bin_val = torch.cat((F.relu(1.0 - torch.abs(bin_diff[:, :-1, :, :]), inplace=True), (1.0 + bin_diff[:, -1:, :, :]).clamp(0, 1)), dim=1)
        return bin_val


class InstanceL2Norm(nn.Module):
    """Instance L2 normalization.
    """

    def __init__(self, size_average=True, eps=1e-05, scale=1.0):
        super().__init__()
        self.size_average = size_average
        self.eps = eps
        self.scale = scale

    def forward(self, input):
        if self.size_average:
            return input * (self.scale * (input.shape[1] * input.shape[2] * input.shape[3] / (torch.sum((input * input).view(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps)).sqrt())
        else:
            return input * (self.scale / (torch.sum((input * input).view(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps).sqrt())


def interpolate(t, sz, mode='bilinear'):
    sz = sz.tolist() if torch.is_tensor(sz) else sz
    align = {} if mode == 'nearest' else dict(align_corners=False)
    return F.interpolate(t, sz, mode=mode, **align) if t.shape[-2:] != sz else t


class InterpCat(nn.Module):
    """Interpolate and concatenate features of different resolutions."""

    def forward(self, input):
        if isinstance(input, (dict, OrderedDict)):
            input = list(input.values())
        output_shape = None
        for x in input:
            if output_shape is None or output_shape[0] > x.shape[-2]:
                output_shape = x.shape[-2:]
        return torch.cat([interpolate(x, output_shape) for x in input], dim=-3)


class GIoULoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weights=None):
        if pred.dim() == 4:
            pred = pred.unsqueeze(0)
        pred = pred.permute(0, 1, 3, 4, 2).reshape(-1, 4)
        target = target.permute(0, 1, 3, 4, 2).reshape(-1, 4)
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]
        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]
        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_union = g_w_intersect * g_h_intersect + 1e-07
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect + 1e-07
        ious = area_intersect / area_union
        gious = ious - (ac_union - area_union) / ac_union
        losses = 1 - gious
        if weights is not None and weights.sum() > 0:
            weights = weights.permute(0, 1, 3, 4, 2).reshape(-1)
            loss_mean = losses[weights > 0].mean()
            ious = ious[weights > 0]
        else:
            loss_mean = losses.mean()
        return loss_mean, ious


class KLRegression(nn.Module):
    """KL-divergence loss for probabilistic regression.
    It is computed using Monte Carlo (MC) samples from an arbitrary distribution."""

    def __init__(self, eps=0.0):
        super().__init__()
        self.eps = eps

    def forward(self, scores, sample_density, gt_density, mc_dim=-1):
        """Args:
            scores: predicted score values
            sample_density: probability density of the sample distribution
            gt_density: probability density of the ground truth distribution
            mc_dim: dimension of the MC samples"""
        exp_val = scores - torch.log(sample_density + self.eps)
        L = torch.logsumexp(exp_val, dim=mc_dim) - math.log(scores.shape[mc_dim]) - torch.mean(scores * (gt_density / (sample_density + self.eps)), dim=mc_dim)
        return L.mean()


class MLRegression(nn.Module):
    """Maximum likelihood loss for probabilistic regression.
    It is computed using Monte Carlo (MC) samples from an arbitrary distribution."""

    def __init__(self, eps=0.0):
        super().__init__()
        self.eps = eps

    def forward(self, scores, sample_density, gt_density=None, mc_dim=-1):
        """Args:
            scores: predicted score values. First sample must be ground-truth
            sample_density: probability density of the sample distribution
            gt_density: not used
            mc_dim: dimension of the MC samples. Only mc_dim=1 supported"""
        assert mc_dim == 1
        assert (sample_density[:, 0, ...] == -1).all()
        exp_val = scores[:, 1:, ...] - torch.log(sample_density[:, 1:, ...] + self.eps)
        L = torch.logsumexp(exp_val, dim=mc_dim) - math.log(scores.shape[mc_dim] - 1) - scores[:, 0, ...]
        loss = L.mean()
        return loss


class KLRegressionGrid(nn.Module):
    """KL-divergence loss for probabilistic regression.
    It is computed using the grid integration strategy."""

    def forward(self, scores, gt_density, grid_dim=-1, grid_scale=1.0):
        """Args:
            scores: predicted score values
            gt_density: probability density of the ground truth distribution
            grid_dim: dimension(s) of the grid
            grid_scale: area of one grid cell"""
        score_corr = grid_scale * torch.sum(scores * gt_density, dim=grid_dim)
        L = torch.logsumexp(scores, dim=grid_dim) + math.log(grid_scale) - score_corr
        return L.mean()


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\\infty and +\\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
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


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\\infty and +\\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)) for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


class LovaszHingeWithLogitsLoss(nn.Module):

    def __init__(self, per_image):
        super(LovaszHingeWithLogitsLoss, self).__init__()
        self.per_image = per_image

    def forward(self, input, target):
        return lovasz_hinge(input, target, per_image=self.per_image)


class StableBCELoss(torch.nn.modules.Module):

    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class LovaszSegLoss(nn.Module):

    def __init__(self, classes=[1], per_image=True):
        super().__init__()
        self.classes = classes
        self.per_image = per_image

    def forward(self, input, target):
        return lovasz_loss.lovasz_softmax(probas=torch.sigmoid(input), labels=target, per_image=self.per_image, classes=self.classes)


def precision(m, gt_m):
    mask = ((m > -1) & (gt_m >= -1)).float()
    prec = ((m == gt_m) * mask).sum(1) / torch.max(mask.sum(1), torch.ones_like(mask.sum(1)))
    no_match_mask = (gt_m > -1).sum(1) == 0
    prec[no_match_mask] = float('NaN')
    return prec


def recall(m, gt_m):
    mask = (gt_m > -1).float()
    return ((m == gt_m) * mask).sum(1) / mask.sum(1)


class TargetCandidateMatchingLoss(nn.Module):

    def __init__(self, nll_balancing=0.5, nll_weight=1.0):
        super().__init__()
        self.nll_balancing = nll_balancing
        self.nll_weight = nll_weight

    def metrics(self, matches1, gt_matches1, **kwargs):
        rec = recall(matches1, gt_matches1[0])
        prec = precision(matches1, gt_matches1[0])
        return {'match_recall': rec, 'match_precision': prec}

    def forward(self, gt_assignment, gt_matches0, gt_matches1, log_assignment, bin_score, **kwargs):
        gt_assignment = gt_assignment[0]
        gt_matches0 = gt_matches0[0]
        gt_matches1 = gt_matches1[0]
        losses = {'total': 0}
        positive = gt_assignment.float()
        neg0 = (gt_matches0 == -1).float()
        neg1 = (gt_matches1 == -1).float()
        num_pos = torch.max(positive.sum((1, 2)), positive.new_tensor(1))
        num_neg = torch.max(neg0.sum(1) + neg1.sum(1), neg0.new_tensor(1))
        nll_pos = -(log_assignment[:, :-1, :-1] * positive).sum((1, 2))
        nll_pos /= num_pos
        nll_neg0 = -(log_assignment[:, :-1, -1] * neg0).sum(1)
        nll_neg1 = -(log_assignment[:, -1, :-1] * neg1).sum(1)
        nll_neg = (nll_neg0 + nll_neg1) / num_neg
        nll = self.nll_balancing * nll_pos + (1 - self.nll_balancing) * nll_neg
        losses['assignment_nll'] = nll
        if self.nll_weight > 0:
            losses['total'] = nll * self.nll_weight
        losses['nll_pos'] = nll_pos
        losses['nll_neg'] = nll_neg
        losses['num_matchable'] = num_pos
        losses['num_unmatchable'] = num_neg
        losses['sinkhorn_norm'] = log_assignment.exp()[:, :-1].sum(2).mean(1)
        losses['bin_score'] = bin_score[None]
        return losses


class LBHinge(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None):
        super().__init__()
        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100
        self.clip = clip

    def forward(self, prediction, label, target_bb=None):
        negative_mask = (label < self.threshold).float()
        positive_mask = 1.0 - negative_mask
        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction
        loss = self.error_metric(prediction, positive_mask * label)
        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
        return loss


class LBHingev2(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=None, threshold=None, return_per_sequence=False):
        super().__init__()
        if error_metric is None:
            if return_per_sequence:
                reduction = 'none'
            else:
                reduction = 'mean'
            error_metric = nn.MSELoss(reduction=reduction)
        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100
        self.return_per_sequence = return_per_sequence

    def forward(self, prediction, label, target_bb=None, valid_samples=None):
        assert prediction.dim() == 4 and label.dim() == 4
        negative_mask = (label < self.threshold).float()
        positive_mask = 1.0 - negative_mask
        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction
        if valid_samples is not None:
            valid_samples = valid_samples.float()
            prediction = prediction * valid_samples
            label = label * valid_samples
            loss = self.error_metric(prediction, positive_mask * label)
            if self.return_per_sequence:
                loss = loss.mean((-2, -1))
            else:
                loss = loss * valid_samples.numel() / (valid_samples.sum() + 1e-12)
        else:
            loss = self.error_metric(prediction, positive_mask * label)
            if self.return_per_sequence:
                loss = loss.mean((-2, -1))
        return loss


class IsTargetCellLoss(nn.Module):

    def __init__(self, return_per_sequence=False, use_with_logits=True):
        super(IsTargetCellLoss, self).__init__()
        self.return_per_sequence = return_per_sequence
        self.use_with_logits = use_with_logits

    def forward(self, prediction, label, target_bb=None, valid_samples=None):
        score_shape = label.shape[-2:]
        prediction = prediction.view(-1, score_shape[0], score_shape[1])
        label = label.view(-1, score_shape[0], score_shape[1])
        if valid_samples is not None:
            valid_samples = valid_samples.float().view(-1)
            if self.use_with_logits:
                prediction_accuracy_persample = F.binary_cross_entropy_with_logits(prediction, label, reduction='none')
            else:
                prediction_accuracy_persample = F.binary_cross_entropy(prediction, label, reduction='none')
            prediction_accuracy = prediction_accuracy_persample.mean((-2, -1))
            prediction_accuracy = prediction_accuracy * valid_samples
            if not self.return_per_sequence:
                num_valid_samples = valid_samples.sum()
                if num_valid_samples > 0:
                    prediction_accuracy = prediction_accuracy.sum() / num_valid_samples
                else:
                    prediction_accuracy = 0.0 * prediction_accuracy.sum()
        else:
            if self.use_with_logits:
                prediction_accuracy = F.binary_cross_entropy_with_logits(prediction, label, reduction='none')
            else:
                prediction_accuracy = F.binary_cross_entropy(prediction, label, reduction='none')
            if self.return_per_sequence:
                prediction_accuracy = prediction_accuracy.mean((-2, -1))
            else:
                prediction_accuracy = prediction_accuracy.mean()
        return prediction_accuracy


class TrackingClassificationAccuracy(nn.Module):
    """ Estimates tracking accuracy by computing whether the peak of the predicted score map matches with the target
        location.
    """

    def __init__(self, threshold, neg_threshold=None):
        super(TrackingClassificationAccuracy, self).__init__()
        self.threshold = threshold
        if neg_threshold is None:
            neg_threshold = threshold
        self.neg_threshold = neg_threshold

    def forward(self, prediction, label, valid_samples=None):
        prediction_reshaped = prediction.view(-1, prediction.shape[-2] * prediction.shape[-1])
        label_reshaped = label.view(-1, label.shape[-2] * label.shape[-1])
        prediction_max_val, argmax_id = prediction_reshaped.max(dim=1)
        label_max_val, _ = label_reshaped.max(dim=1)
        label_val_at_peak = label_reshaped[torch.arange(len(argmax_id)), argmax_id]
        label_val_at_peak = torch.max(label_val_at_peak, torch.zeros_like(label_val_at_peak))
        prediction_correct = (label_val_at_peak >= self.threshold) & (label_max_val > 0.25) | (label_val_at_peak < self.neg_threshold) & (label_max_val < 0.25)
        if valid_samples is not None:
            valid_samples = valid_samples.float().view(-1)
            num_valid_samples = valid_samples.sum()
            if num_valid_samples > 0:
                prediction_accuracy = (valid_samples * prediction_correct.float()).sum() / num_valid_samples
            else:
                prediction_accuracy = 1.0
        else:
            prediction_accuracy = prediction_correct.float().mean()
        return prediction_accuracy, prediction_correct.float()


def adaptive_cat(seq, dim=0, ref_tensor=0, mode='bilinear'):
    sz = seq[ref_tensor].shape[-2:]
    t = torch.cat([interpolate(t, sz, mode=mode) for t in seq], dim=dim)
    return t


def relu(negative_slope=0.0, inplace=False):
    return nn.LeakyReLU(negative_slope, inplace=inplace)


class TSE(nn.Module):

    def __init__(self, fc, ic, oc):
        super().__init__()
        self.reduce = nn.Sequential(conv(fc, oc, 1), relu(), conv(oc, oc, 1))
        nc = ic + oc
        self.transform = nn.Sequential(conv(nc, nc, 3), relu(), conv(nc, nc, 3), relu(), conv(nc, oc, 3), relu())

    def forward(self, ft, score, x=None):
        h = self.reduce(ft)
        hpool = F.adaptive_avg_pool2d(h, (1, 1)) if x is None else x
        h = adaptive_cat((h, score), dim=1, ref_tensor=0)
        h = self.transform(h)
        return h, hpool


class CAB(nn.Module):

    def __init__(self, oc, deepest):
        super().__init__()
        self.convreluconv = nn.Sequential(conv(2 * oc, oc, 1), relu(), conv(oc, oc, 1))
        self.deepest = deepest

    def forward(self, deeper, shallower, att_vec=None):
        shallow_pool = F.adaptive_avg_pool2d(shallower, (1, 1))
        deeper_pool = deeper if self.deepest else F.adaptive_avg_pool2d(deeper, (1, 1))
        if att_vec is not None:
            global_pool = torch.cat([shallow_pool, deeper_pool, att_vec], dim=1)
        else:
            global_pool = torch.cat((shallow_pool, deeper_pool), dim=1)
        conv_1x1 = self.convreluconv(global_pool)
        inputs = shallower * torch.sigmoid(conv_1x1)
        out = inputs + interpolate(deeper, inputs.shape[-2:])
        return out


class RRB(nn.Module):

    def __init__(self, oc, use_bn=False):
        super().__init__()
        self.conv1x1 = conv(oc, oc, 1)
        if use_bn:
            self.bblock = nn.Sequential(conv(oc, oc, 3), nn.BatchNorm2d(oc), relu(), conv(oc, oc, 3, bias=False))
        else:
            self.bblock = nn.Sequential(conv(oc, oc, 3), relu(), conv(oc, oc, 3, bias=False))

    def forward(self, x):
        h = self.conv1x1(x)
        return F.relu(h + self.bblock(h))


class Upsampler(nn.Module):

    def __init__(self, in_channels=64):
        super().__init__()
        self.conv1 = conv(in_channels, in_channels // 2, 3)
        self.conv2 = conv(in_channels // 2, 1, 3)

    def forward(self, x, image_size):
        x = F.interpolate(x, (2 * x.shape[-2], 2 * x.shape[-1]), mode='bicubic', align_corners=False)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, image_size[-2:], mode='bicubic', align_corners=False)
        x = self.conv2(x)
        return x


class LWTLDecoder(nn.Module):
    """ Decoder module """

    def __init__(self, in_channels=1, out_channels=32, ft_channels=None, use_bn=False):
        super().__init__()
        assert ft_channels is not None
        self.ft_channels = ft_channels
        self.TSE = nn.ModuleDict()
        self.RRB1 = nn.ModuleDict()
        self.CAB = nn.ModuleDict()
        self.RRB2 = nn.ModuleDict()
        self.proj = nn.ModuleDict()
        ic = in_channels
        oc = {'layer1': 1, 'layer2': 2, 'layer3': 2, 'layer4': 4}
        out_feature_channels = {}
        if 'layer4' in ft_channels.keys():
            last_layer = 'layer4'
        else:
            last_layer = 'layer3'
        prev_layer = None
        for L, fc in self.ft_channels.items():
            if not L == last_layer:
                self.proj[L] = nn.Sequential(conv(oc[prev_layer] * out_channels, oc[L] * out_channels, 1), relu())
            self.TSE[L] = TSE(fc, ic, oc[L] * out_channels)
            self.RRB1[L] = RRB(oc[L] * out_channels, use_bn=use_bn)
            self.CAB[L] = CAB(oc[L] * out_channels, L == last_layer)
            self.RRB2[L] = RRB(oc[L] * out_channels, use_bn=use_bn)
            out_feature_channels['{}_dec'.format(L)] = oc[L] * out_channels
            prev_layer = L
        self.project = Upsampler(out_channels)
        self._out_feature_channels = out_feature_channels

    def out_feature_channels(self):
        return self._out_feature_channels

    def forward(self, scores, features, image_size, output_layers=(), num_objects=None):
        if num_objects is None:
            assert scores.dim() == 5
        else:
            assert scores.dim() == 6
        outputs = OrderedDict()
        scores = scores.view(-1, *scores.shape[-3:])
        x = None
        for i, L in enumerate(self.ft_channels):
            ft = features[L]
            s = interpolate(scores, ft.shape[-2:])
            if not x is None:
                x = self.proj[L](x)
            if num_objects is not None:
                h, hpool = self.TSE[L](ft.view(ft.shape[0], 1, *ft.shape[-3:]).repeat(1, num_objects, 1, 1, 1).view(-1, *ft.shape[-3:]), s, x)
            else:
                h, hpool = self.TSE[L](ft, s, x)
            h = self.RRB1[L](h)
            h = self.CAB[L](hpool, h)
            x = self.RRB2[L](h)
            if '{}_dec'.format(L) in output_layers:
                outputs['{}_dec'.format(L)] = x
        x = self.project(x, image_size)
        return x, outputs


class FilterInitializerZero(nn.Module):
    """Initializes a target classification filter with zeros.
    args:
        filter_size:  Size of the filter.
        feature_dim:  Input feature dimentionality."""

    def __init__(self, filter_size=1, feature_dim=256):
        super().__init__()
        self.filter_size = feature_dim, filter_size, filter_size

    def forward(self, feat, bb):
        """Runs the initializer module.
        Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
        returns:
            weights:  The output weights. Dims (sequences, feat_dim, wH, wW)."""
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        return feat.new_zeros(num_sequences, self.filter_size[0], self.filter_size[1], self.filter_size[2])


class ResidualDS16SW(nn.Module):
    """ Outputs the few-shot learner label and spatial importance weights given the segmentation mask """

    def __init__(self, layer_dims, use_bn=True):
        super().__init__()
        self.conv_block = conv_block(1, layer_dims[0], kernel_size=3, stride=2, padding=1, batch_norm=use_bn)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1, use_bn=use_bn)
        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2, use_bn=use_bn)
        self.label_pred = conv_block(layer_dims[2], layer_dims[3], kernel_size=3, stride=1, padding=1, relu=True, batch_norm=use_bn)
        self.samp_w_pred = nn.Conv2d(layer_dims[2], layer_dims[3], kernel_size=3, padding=1, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.samp_w_pred.weight.data.fill_(0)
        self.samp_w_pred.bias.data.fill_(1)

    def forward(self, label_mask, feature=None):
        assert label_mask.dim() == 4
        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])
        out = self.pool(self.conv_block(label_mask))
        out = self.res2(self.res1(out))
        label_enc = self.label_pred(out)
        sample_w = self.samp_w_pred(out)
        label_enc = label_enc.view(label_shape[0], label_shape[1], *label_enc.shape[-3:])
        sample_w = sample_w.view(label_shape[0], label_shape[1], *sample_w.shape[-3:])
        return label_enc, sample_w


class ResidualDS16FeatSWBoxCatMultiBlock(nn.Module):

    def __init__(self, layer_dims, feat_dim, use_final_relu=True, use_gauss=True, use_bn=True, non_default_init=True, init_bn=1, gauss_scale=0.25, final_bn=True):
        super().__init__()
        in_layer_dim = (feat_dim + 1,) + tuple(list(layer_dims)[:-2])
        out_layer_dim = tuple(list(layer_dims)[:-1])
        self.use_gauss = use_gauss
        res = []
        for in_d, out_d in zip(in_layer_dim, out_layer_dim):
            ds = nn.Conv2d(in_d, out_d, kernel_size=3, padding=1, stride=1)
            res.append(BasicBlock(in_d, out_d, stride=1, downsample=ds, use_bn=use_bn))
        self.res = nn.Sequential(*res)
        self.label_pred = conv_block(layer_dims[-2], layer_dims[-1], kernel_size=3, stride=1, padding=1, relu=use_final_relu, batch_norm=final_bn)
        self.gauss_scale = gauss_scale
        if non_default_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(init_bn)
                    m.bias.data.zero_()

    def bbox_to_mask(self, bbox, sz):
        mask = torch.zeros((bbox.shape[0], 1, *sz), dtype=torch.float32, device=bbox.device)
        for i, bb in enumerate(bbox):
            x1, y1, w, h = list(map(int, bb))
            x1 = int(x1 + 0.5)
            y1 = int(y1 + 0.5)
            h = int(h + 0.5)
            w = int(w + 0.5)
            mask[i, :, y1:y1 + h, x1:x1 + w] = 1.0
        return mask

    def bbox_to_gauss(self, bbox, sz):
        mask = torch.zeros((bbox.shape[0], 1, *sz), dtype=torch.float32, device=bbox.device)
        x_max, y_max = sz[-1], sz[-2]
        for i, bb in enumerate(bbox):
            x1, y1, w, h = list(map(int, bb))
            cx, cy = x1 + w / 2, y1 + h / 2
            xcoords = torch.arange(0, x_max).unsqueeze(dim=0).float()
            ycoords = torch.arange(0, y_max).unsqueeze(dim=0).T.float()
            d_xcoords = xcoords - cx
            d_ycoords = ycoords - cy
            dtotsqr = d_xcoords ** 2 / (self.gauss_scale * w) ** 2 + d_ycoords ** 2 / (self.gauss_scale * h) ** 2
            mask[i, 0] = torch.exp(-0.5 * dtotsqr)
        return mask

    def forward(self, bb, feat, sz):
        if self.use_gauss:
            label_mask = self.bbox_to_gauss(bb, sz[-2:])
        else:
            label_mask = self.bbox_to_mask(bb, sz[-2:])
        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])
        feat = feat.view(-1, *feat.shape[-3:])
        feat_mask_enc = torch.cat([feat, interpolate(label_mask, feat.shape[-2:])], dim=1)
        out = self.res(feat_mask_enc)
        label_enc = self.label_pred(out)
        label_enc = label_enc.view(label_shape[0], label_shape[1], *label_enc.shape[-3:])
        return label_enc


class ResidualDS16FeatSWBox(nn.Module):

    def __init__(self, layer_dims, feat_dim, use_final_relu=True, use_gauss=True, use_bn=False, use_sample_w=True):
        super().__init__()
        self.use_sample_w = use_sample_w
        self.use_gauss = use_gauss
        self.conv_block = conv_block(1, layer_dims[0], kernel_size=3, stride=2, padding=1, batch_norm=use_bn)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1, use_bn=use_bn)
        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2, use_bn=use_bn)
        ds3 = nn.Conv2d(layer_dims[2] + feat_dim, layer_dims[3], kernel_size=3, padding=1, stride=1)
        self.res3 = BasicBlock(layer_dims[2] + feat_dim, layer_dims[3], stride=1, downsample=ds3, use_bn=use_bn)
        self.label_pred = conv_block(layer_dims[3], layer_dims[4], kernel_size=3, stride=1, padding=1, relu=use_final_relu)
        if self.use_sample_w:
            self.samp_w_pred = nn.Conv2d(layer_dims[3], layer_dims[4], kernel_size=3, padding=1, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.use_sample_w:
            self.samp_w_pred.weight.data.fill_(0)
            self.samp_w_pred.bias.data.fill_(1)

    def bbox_to_mask(self, bbox, sz):
        mask = torch.zeros((bbox.shape[0], 1, *sz), dtype=torch.float32, device=bbox.device)
        for i, bb in enumerate(bbox):
            x1, y1, w, h = list(map(int, bb))
            x1 = int(x1 + 0.5)
            y1 = int(y1 + 0.5)
            h = int(h + 0.5)
            w = int(w + 0.5)
            mask[i, :, max(0, y1):y1 + h, max(0, x1):x1 + w] = 1.0
        return mask

    def bbox_to_gauss(self, bbox, sz):
        mask = torch.zeros((bbox.shape[0], 1, *sz), dtype=torch.float32, device=bbox.device)
        x_max, y_max = sz[-1], sz[-2]
        for i, bb in enumerate(bbox):
            x1, y1, w, h = list(map(int, bb))
            cx, cy = x1 + w / 2, y1 + h / 2
            xcoords = torch.arange(0, x_max).unsqueeze(dim=0).float()
            ycoords = torch.arange(0, y_max).unsqueeze(dim=0).T.float()
            d_xcoords = xcoords - cx
            d_ycoords = ycoords - cy
            dtotsqr = d_xcoords ** 2 / (0.25 * w) ** 2 + d_ycoords ** 2 / (0.25 * h) ** 2
            mask[i, 0] = torch.exp(-0.5 * dtotsqr)
        return mask

    def forward(self, bb, feat, sz):
        assert bb.dim() == 3
        num_frames = bb.shape[0]
        batch_sz = bb.shape[1]
        bb = bb.reshape(-1, 4)
        if self.use_gauss:
            label_mask = self.bbox_to_gauss(bb, sz[-2:])
        else:
            label_mask = self.bbox_to_mask(bb, sz[-2:])
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])
        mask_enc = self.pool(self.conv_block(label_mask))
        mask_enc = self.res2(self.res1(mask_enc))
        feat = feat.view(-1, *feat.shape[-3:])
        feat_mask_enc = torch.cat((mask_enc, feat), dim=1)
        out = self.res3(feat_mask_enc)
        label_enc = self.label_pred(out)
        label_enc = label_enc.view(num_frames, batch_sz, *label_enc.shape[-3:])
        sample_w = None
        if self.use_sample_w:
            sample_w = self.samp_w_pred(out)
            sample_w = sample_w.view(num_frames, batch_sz, *sample_w.shape[-3:])
        return label_enc, sample_w


class LinearFilter(nn.Module):
    """Target classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features."""

    def __init__(self, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None):
        super().__init__()
        self.filter_size = filter_size
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor
        if self.feature_extractor:
            for m in self.feature_extractor.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, train_feat, test_feat, train_bb, *args, **kwargs):
        """Learns a target classification filter based on the train samples and return the resulting classification
        scores on the test samples.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_feat:  Backbone features for the train samples (4 or 5 dims).
            test_feat:  Backbone features for the test samples (4 or 5 dims).
            train_bb:  Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            test_scores:  Classification scores on the test samples."""
        assert train_bb.dim() == 3
        num_sequences = train_bb.shape[1]
        train_feat = self.extract_classification_feat(train_feat, num_sequences)
        test_feat = self.extract_classification_feat(test_feat, num_sequences)
        filter, filter_iter, losses = self.get_filter(train_feat, train_bb, *args, **kwargs)
        test_scores = [self.classify(f, test_feat) for f in filter_iter]
        return test_scores

    def extract_classification_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)
        if feat.dim() == 5:
            feat = feat.reshape(-1, *feat.shape[-3:])
        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""
        scores = filter_layer.apply_filter(feat, weights)
        return scores

    def get_filter(self, feat, bb, *args, **kwargs):
        """Outputs the learned filter based on the input features (feat) and target boxes (bb) by running the
        filter initializer and optimizer. Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            weights:  The final oprimized weights. Dims (sequences, feat_dim, wH, wW).
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""
        weights = self.filter_initializer(feat, bb)
        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(weights, *args, feat=feat, bb=bb, **kwargs)
        else:
            weights_iter = [weights]
            losses = None
        return weights, weights_iter, losses

    def train_classifier(self, backbone_feat, bb):
        num_sequences = bb.shape[1]
        if backbone_feat.dim() == 5:
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:])
        train_feat = self.extract_classification_feat(backbone_feat, num_sequences)
        final_filter, _, train_losses = self.get_filter(train_feat, bb)
        return final_filter, train_losses

    def track_frame(self, filter_weights, backbone_feat):
        if backbone_feat.dim() == 5:
            num_sequences = backbone_feat.shape[1]
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:])
        else:
            num_sequences = None
        test_feat = self.extract_classification_feat(backbone_feat, num_sequences)
        scores = filter_layer.apply_filter(test_feat, filter_weights)
        return scores


class TensorList(list):
    """Container mainly used for lists of torch tensors. Extends lists with pytorch functionality."""

    def __init__(self, list_of_tensors=None):
        if list_of_tensors is None:
            list_of_tensors = list()
        super(TensorList, self).__init__(list_of_tensors)

    def __deepcopy__(self, memodict={}):
        return TensorList(copy.deepcopy(list(self), memodict))

    def __getitem__(self, item):
        if isinstance(item, int):
            return super(TensorList, self).__getitem__(item)
        elif isinstance(item, (tuple, list)):
            return TensorList([super(TensorList, self).__getitem__(i) for i in item])
        else:
            return TensorList(super(TensorList, self).__getitem__(item))

    def __add__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e1 + e2) for e1, e2 in zip(self, other)])
        return TensorList([(e + other) for e in self])

    def __radd__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e2 + e1) for e1, e2 in zip(self, other)])
        return TensorList([(other + e) for e in self])

    def __iadd__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] += e2
        else:
            for i in range(len(self)):
                self[i] += other
        return self

    def __sub__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e1 - e2) for e1, e2 in zip(self, other)])
        return TensorList([(e - other) for e in self])

    def __rsub__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e2 - e1) for e1, e2 in zip(self, other)])
        return TensorList([(other - e) for e in self])

    def __isub__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] -= e2
        else:
            for i in range(len(self)):
                self[i] -= other
        return self

    def __mul__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e1 * e2) for e1, e2 in zip(self, other)])
        return TensorList([(e * other) for e in self])

    def __rmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e2 * e1) for e1, e2 in zip(self, other)])
        return TensorList([(other * e) for e in self])

    def __imul__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] *= e2
        else:
            for i in range(len(self)):
                self[i] *= other
        return self

    def __truediv__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e1 / e2) for e1, e2 in zip(self, other)])
        return TensorList([(e / other) for e in self])

    def __rtruediv__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e2 / e1) for e1, e2 in zip(self, other)])
        return TensorList([(other / e) for e in self])

    def __itruediv__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] /= e2
        else:
            for i in range(len(self)):
                self[i] /= other
        return self

    def __matmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e1 @ e2) for e1, e2 in zip(self, other)])
        return TensorList([(e @ other) for e in self])

    def __rmatmul__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e2 @ e1) for e1, e2 in zip(self, other)])
        return TensorList([(other @ e) for e in self])

    def __imatmul__(self, other):
        if TensorList._iterable(other):
            for i, e2 in enumerate(other):
                self[i] @= e2
        else:
            for i in range(len(self)):
                self[i] @= other
        return self

    def __mod__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e1 % e2) for e1, e2 in zip(self, other)])
        return TensorList([(e % other) for e in self])

    def __rmod__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e2 % e1) for e1, e2 in zip(self, other)])
        return TensorList([(other % e) for e in self])

    def __pos__(self):
        return TensorList([(+e) for e in self])

    def __neg__(self):
        return TensorList([(-e) for e in self])

    def __le__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e1 <= e2) for e1, e2 in zip(self, other)])
        return TensorList([(e <= other) for e in self])

    def __ge__(self, other):
        if TensorList._iterable(other):
            return TensorList([(e1 >= e2) for e1, e2 in zip(self, other)])
        return TensorList([(e >= other) for e in self])

    def concat(self, other):
        return TensorList(super(TensorList, self).__add__(other))

    def copy(self):
        return TensorList(super(TensorList, self).copy())

    def unroll(self):
        if not any(isinstance(t, TensorList) for t in self):
            return self
        new_list = TensorList()
        for t in self:
            if isinstance(t, TensorList):
                new_list.extend(t.unroll())
            else:
                new_list.append(t)
        return new_list

    def list(self):
        return list(self)

    def attribute(self, attr: str, *args):
        return TensorList([getattr(e, attr, *args) for e in self])

    def apply(self, fn):
        return TensorList([fn(e) for e in self])

    def __getattr__(self, name):
        if not hasattr(torch.Tensor, name):
            raise AttributeError("'TensorList' object has not attribute '{}'".format(name))

        def apply_attr(*args, **kwargs):
            return TensorList([getattr(e, name)(*args, **kwargs) for e in self])
        return apply_attr

    @staticmethod
    def _iterable(a):
        return isinstance(a, (TensorList, list))


class LWTLResidual(nn.Module):
    """ Computes the residuals W(y_t)*(T_tau(x_t) - E(y_t) and lambda*tau in the few-shot learner loss (3) in the
    paper """

    def __init__(self, init_filter_reg=0.01, filter_dilation_factors=None):
        super().__init__()
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.filter_dilation_factors = filter_dilation_factors

    def forward(self, meta_parameter: TensorList, feat, label, sample_weight=None):
        filter = meta_parameter[0]
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        scores = filter_layer.apply_filter(feat, filter, dilation_factors=self.filter_dilation_factors)
        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images)
        elif isinstance(sample_weight, torch.Tensor):
            if sample_weight.numel() == scores.numel():
                sample_weight = sample_weight.view(scores.shape)
            elif sample_weight.dim() == 1:
                sample_weight = sample_weight.view(-1, 1, 1, 1, 1)
        label = label.view(scores.shape)
        data_residual = sample_weight * (scores - label)
        reg_residual = self.filter_reg * filter.view(1, num_sequences, -1)
        return TensorList([data_residual, reg_residual])


class LWTLBoxNet(nn.Module):

    def __init__(self, feature_extractor, target_model, decoder, target_model_input_layer, decoder_input_layers, label_encoder=None, box_label_encoder=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.target_model = target_model
        self.decoder = decoder
        self.label_encoder = label_encoder
        self.target_model_input_layer = (target_model_input_layer,) if isinstance(target_model_input_layer, str) else target_model_input_layer
        self.decoder_input_layers = decoder_input_layers
        self.output_layers = sorted(list(set(self.target_model_input_layer + self.decoder_input_layers)))
        self.box_label_encoder = box_label_encoder
        self.train_only_box_label_gen = True

    def train(self, mode=True):
        for x in self.feature_extractor.parameters():
            x.requires_grad_(False)
        self.feature_extractor.eval()
        if mode:
            for x in self.box_label_encoder.parameters():
                x.requires_grad_(True)
            self.box_label_encoder.train()
            if self.train_only_box_label_gen:
                for x in self.target_model.parameters():
                    x.requires_grad_(False)
                self.target_model.eval()
                for x in self.label_encoder.parameters():
                    x.requires_grad_(False)
                self.label_encoder.eval()
                for x in self.decoder.parameters():
                    x.requires_grad_(False)
                self.decoder.eval()
            else:
                for x in self.target_model.parameters():
                    x.requires_grad_(True)
                self.target_model.train()
                for x in self.label_encoder.parameters():
                    x.requires_grad_(True)
                self.label_encoder.train()
                for x in self.decoder.parameters():
                    x.requires_grad_(True)
                self.decoder.train()
        else:
            for x in self.target_model.parameters():
                x.requires_grad_(False)
            self.target_model.eval()
            for x in self.label_encoder.parameters():
                x.requires_grad_(False)
            self.label_encoder.eval()
            for x in self.decoder.parameters():
                x.requires_grad_(False)
            self.decoder.eval()
            for x in self.box_label_encoder.parameters():
                x.requires_grad_(False)
            self.box_label_encoder.eval()

    def forward(self, train_imgs, test_imgs, train_masks, test_masks, bb_train, num_refinement_iter=2):
        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
        assert train_masks.dim() == 4, 'Expect 4 dimensional masks'
        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]
        num_test_frames = test_imgs.shape[0]
        train_feat = self.extract_backbone_features(train_imgs.contiguous().view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = self.extract_backbone_features(test_imgs.contiguous().view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))
        train_feat_clf = self.extract_classification_feat(train_feat)
        test_feat_clf = self.extract_classification_feat(test_feat)
        bb_mask_enc = self.box_label_encoder(bb_train, train_feat_clf)
        box_mask_pred, decoder_feat = self.decoder(bb_mask_enc, test_feat, test_imgs.shape[-2:], ('layer4_dec', 'layer3_dec', 'layer2_dec', 'layer1_dec'))
        mask_enc = self.label_encoder(box_mask_pred, train_feat_clf)
        mask_enc_test = self.label_encoder(test_masks.contiguous(), test_feat_clf)
        train_feat_clf = train_feat_clf.view(num_train_frames, num_sequences, *train_feat_clf.shape[-3:])
        filter, filter_iter, _ = self.target_model.get_filter(train_feat_clf, *mask_enc)
        test_feat_clf = test_feat_clf.view(num_test_frames, num_sequences, *test_feat_clf.shape[-3:])
        target_scores = [self.target_model.classify(f, test_feat_clf) for f in filter_iter]
        target_scores_last_iter = target_scores[-1]
        mask_pred, decoder_feat = self.decoder(target_scores_last_iter, test_feat, test_imgs.shape[-2:], ('layer4_dec', 'layer3_dec', 'layer2_dec', 'layer1_dec'))
        decoder_feat['mask_enc'] = target_scores_last_iter.view(-1, *target_scores_last_iter.shape[-3:])
        if isinstance(mask_enc_test, (tuple, list)):
            mask_enc_test = mask_enc_test[0]
        return mask_pred, target_scores, mask_enc_test, box_mask_pred

    def segment_target(self, target_filter, test_feat_tm, test_feat):
        assert target_filter.dim() == 5
        test_feat_tm = test_feat_tm.view(1, 1, *test_feat_tm.shape[-3:])
        mask_encoding_pred = self.target_model.apply_target_model(target_filter, test_feat_tm)
        mask_pred, decoder_feat = self.decoder(mask_encoding_pred, test_feat, (test_feat_tm.shape[-2] * 16, test_feat_tm.shape[-1] * 16))
        return mask_pred, None

    def get_backbone_target_model_features(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.target_model_input_layer})
        if len(self.target_model_input_layer) == 1:
            return feat[self.target_model_input_layer[0]]
        return feat

    def extract_target_model_features(self, backbone_feat):
        return self.target_model.extract_target_model_features(self.get_backbone_target_model_features(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)


class LWTLNet(nn.Module):

    def __init__(self, feature_extractor, target_model, decoder, target_model_input_layer, decoder_input_layers, label_encoder=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.target_model = target_model
        self.decoder = decoder
        self.label_encoder = label_encoder
        self.target_model_input_layer = (target_model_input_layer,) if isinstance(target_model_input_layer, str) else target_model_input_layer
        self.decoder_input_layers = decoder_input_layers
        self.output_layers = sorted(list(set(self.target_model_input_layer + self.decoder_input_layers)))

    def forward(self, train_imgs, test_imgs, train_masks, test_masks, num_refinement_iter=2):
        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]
        num_test_frames = test_imgs.shape[0]
        train_feat_backbone = self.extract_backbone_features(train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat_backbone = self.extract_backbone_features(test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))
        train_feat_tm = self.extract_target_model_features(train_feat_backbone)
        train_feat_tm = train_feat_tm.view(num_train_frames, num_sequences, *train_feat_tm.shape[-3:])
        train_feat_tm_all = [train_feat_tm]
        few_shot_label, few_shot_sw = self.label_encoder(train_masks, train_feat_tm)
        few_shot_label_all = [few_shot_label]
        few_shot_sw_all = None if few_shot_sw is None else [few_shot_sw]
        test_feat_tm = self.extract_target_model_features(test_feat_backbone)
        filter, filter_iter, _ = self.target_model.get_filter(train_feat_tm, few_shot_label, few_shot_sw)
        mask_predictons_all = []
        for i in range(num_test_frames):
            test_feat_tm_it = test_feat_tm.view(num_test_frames, num_sequences, *test_feat_tm.shape[-3:])[i:i + 1, ...]
            mask_encoding_pred = [self.target_model.apply_target_model(f, test_feat_tm_it) for f in filter_iter]
            test_feat_backbone_it = {k: v.view(num_test_frames, num_sequences, *v.shape[-3:])[i, ...] for k, v in test_feat_backbone.items()}
            mask_encoding_pred_last_iter = mask_encoding_pred[-1]
            mask_pred, decoder_feat = self.decoder(mask_encoding_pred_last_iter, test_feat_backbone_it, test_imgs.shape[-2:])
            mask_pred = mask_pred.view(1, num_sequences, *mask_pred.shape[-2:])
            mask_predictons_all.append(mask_pred)
            mask_pred_prob = torch.sigmoid(mask_pred.clone().detach())
            few_shot_label, few_shot_sw = self.label_encoder(mask_pred_prob, test_feat_tm_it)
            few_shot_label_all.append(few_shot_label)
            if few_shot_sw_all is not None:
                few_shot_sw_all.append(few_shot_sw)
            train_feat_tm_all.append(test_feat_tm_it)
            if i < num_test_frames - 1 and num_refinement_iter > 0:
                train_feat_tm_it = torch.cat(train_feat_tm_all, dim=0)
                few_shot_label_it = torch.cat(few_shot_label_all, dim=0)
                if few_shot_sw_all is not None:
                    few_shot_sw_it = torch.cat(few_shot_sw_all, dim=0)
                else:
                    few_shot_sw_it = None
                filter_updated, _, _ = self.target_model.filter_optimizer(TensorList([filter]), feat=train_feat_tm_it, label=few_shot_label_it, sample_weight=few_shot_sw_it, num_iter=num_refinement_iter)
                filter = filter_updated[0]
        mask_predictons_all = torch.cat(mask_predictons_all, dim=0)
        return mask_predictons_all

    def segment_target(self, target_filter, test_feat_tm, test_feat):
        assert target_filter.dim() == 5
        test_feat_tm = test_feat_tm.view(1, 1, *test_feat_tm.shape[-3:])
        mask_encoding_pred = self.target_model.apply_target_model(target_filter, test_feat_tm)
        mask_pred, decoder_feat = self.decoder(mask_encoding_pred, test_feat, (test_feat_tm.shape[-2] * 16, test_feat_tm.shape[-1] * 16))
        return mask_pred, mask_encoding_pred

    def get_backbone_target_model_features(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.target_model_input_layer})
        if len(self.target_model_input_layer) == 1:
            return feat[self.target_model_input_layer[0]]
        return feat

    def extract_target_model_features(self, backbone_feat):
        return self.target_model.extract_target_model_features(self.get_backbone_target_model_features(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)


class STANet(nn.Module):

    def __init__(self, feature_extractor, target_model, target_model_segm, decoder, target_model_input_layer, decoder_input_layers, label_encoder=None, bbox_encoder=None, segm_encoder=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.target_model = target_model
        self.target_model_segm = target_model_segm
        self.decoder = decoder
        self.label_encoder = label_encoder
        self.bbox_encoder = bbox_encoder
        self.segm_encoder = segm_encoder
        self.target_model_input_layer = (target_model_input_layer,) if isinstance(target_model_input_layer, str) else target_model_input_layer
        self.decoder_input_layers = decoder_input_layers
        self.output_layers = sorted(list(set(self.target_model_input_layer + self.decoder_input_layers)))

    def forward(self, train_imgs, train_bbox):
        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        train_feat_clf = self.extract_target_model_features(train_feat)
        train_bbox_enc, _ = self.label_encoder(train_bbox, train_feat_clf, list(train_imgs.shape[-2:]))
        train_mask_enc, train_mask_sw = self.bbox_encoder(train_bbox, train_feat_clf, list(train_imgs.shape[-2:]))
        train_feat_clf = train_feat_clf.view(num_train_frames, num_sequences, *train_feat_clf.shape[-3:])
        _, filter_iter, _ = self.target_model.get_filter(train_feat_clf, train_mask_enc, train_mask_sw)
        target_scores = [self.target_model.apply_target_model(f, train_feat_clf) for f in filter_iter]
        target_scores_last_iter = target_scores[-1]
        coarse_mask = torch.cat((train_bbox_enc, target_scores_last_iter), dim=2)
        pred_all, _ = self.decoder(coarse_mask, train_feat, train_imgs.shape[-2:])
        pred_all = pred_all.view(num_train_frames, num_sequences, *pred_all.shape[-2:])
        train_segm_enc, train_segm_sw = self.segm_encoder(torch.sigmoid(pred_all), train_feat_clf)
        _, filter_iter_segm, _ = self.target_model_segm.get_filter(train_feat_clf, train_segm_enc, train_segm_sw)
        target_scores_segm = [self.target_model_segm.apply_target_model(f, train_feat_clf) for f in filter_iter_segm]
        target_scores_last_iter_segm = target_scores_segm[-1]
        coarse_mask = torch.cat((train_bbox_enc, target_scores_last_iter_segm), dim=2)
        pred_all_segm, _ = self.decoder(coarse_mask, train_feat, train_imgs.shape[-2:])
        pred_all_segm = pred_all_segm.view(num_train_frames, num_sequences, *pred_all_segm.shape[-2:])
        return pred_all, pred_all_segm

    def segment_target_add_bbox_encoder(self, bbox_mask, target_filter, test_feat_clf, test_feat, segm):
        assert target_filter.dim() == 5
        if not segm:
            target_scores = self.target_model.apply_target_model(target_filter, test_feat_clf)
        else:
            target_scores = self.target_model_segm.apply_target_model(target_filter, test_feat_clf)
        target_scores = torch.cat((bbox_mask, target_scores), dim=2)
        mask_pred, decoder_feat = self.decoder(target_scores, test_feat, (test_feat_clf.shape[-2] * 16, test_feat_clf.shape[-1] * 16))
        return mask_pred

    def get_backbone_target_model_features(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.target_model_input_layer})
        if len(self.target_model_input_layer) == 1:
            return feat[self.target_model_input_layer[0]]
        return feat

    def extract_target_model_features(self, backbone_feat):
        return self.target_model.extract_target_model_features(self.get_backbone_target_model_features(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)


class GNSteepestDescent(nn.Module):
    """General module for steepest descent based meta learning."""

    def __init__(self, residual_module, num_iter=1, compute_losses=False, detach_length=float('Inf'), parameter_batch_dim=0, residual_batch_dim=0, steplength_reg=0.0):
        super().__init__()
        self.residual_module = residual_module
        self.num_iter = num_iter
        self.compute_losses = compute_losses
        self.detach_length = detach_length
        self.steplength_reg = steplength_reg
        self._parameter_batch_dim = parameter_batch_dim
        self._residual_batch_dim = residual_batch_dim

    def _sqr_norm(self, x: TensorList, batch_dim=0):
        sum_keep_batch_dim = lambda e: e.sum(dim=[d for d in range(e.dim()) if d != batch_dim])
        return sum((x * x).apply(sum_keep_batch_dim))

    def _compute_loss(self, res):
        return sum((res * res).sum()) / sum(res.numel())

    def forward(self, meta_parameter: TensorList, num_iter=None, *args, **kwargs):
        input_is_list = True
        if not isinstance(meta_parameter, TensorList):
            meta_parameter = TensorList([meta_parameter])
            input_is_list = False
        torch_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        num_iter = self.num_iter if num_iter is None else num_iter
        meta_parameter_iterates = []

        def _add_iterate(meta_par):
            if input_is_list:
                meta_parameter_iterates.append(meta_par)
            else:
                meta_parameter_iterates.append(meta_par[0])
        _add_iterate(meta_parameter)
        losses = []
        for i in range(num_iter):
            if i > 0 and i % self.detach_length == 0:
                meta_parameter = meta_parameter.detach()
            meta_parameter.requires_grad_(True)
            r = self.residual_module(meta_parameter, **kwargs)
            if self.compute_losses:
                losses.append(self._compute_loss(r))
            u = r.clone()
            g = TensorList(torch.autograd.grad(r, meta_parameter, u, create_graph=True))
            h = TensorList(torch.autograd.grad(g, u, g, create_graph=True))
            ip_gg = self._sqr_norm(g, batch_dim=self._parameter_batch_dim)
            ip_hh = self._sqr_norm(h, batch_dim=self._residual_batch_dim)
            alpha = ip_gg / (ip_hh + self.steplength_reg * ip_gg).clamp(1e-08)
            step = g.apply(lambda e: alpha.reshape([(-1 if d == self._parameter_batch_dim else 1) for d in range(e.dim())]) * e)
            meta_parameter = meta_parameter - step
            _add_iterate(meta_parameter)
        if self.compute_losses:
            losses.append(self._compute_loss(self.residual_module(meta_parameter, **kwargs)))
        torch.set_grad_enabled(torch_grad_enabled)
        if not torch_grad_enabled:
            meta_parameter.detach_()
            for w in meta_parameter_iterates:
                w.detach_()
            for l in losses:
                l.detach_()
        if not input_is_list:
            meta_parameter = meta_parameter[0]
        return meta_parameter, meta_parameter_iterates, losses


class KLRegSteepestDescent(nn.Module):
    """General meta learning module for Steepest Descent based meta learning with Newton when minimizing KL-divergence."""

    def __init__(self, score_predictor, num_iter=1, compute_losses=True, detach_length=float('Inf'), parameter_batch_dim=0, steplength_reg=0.0, hessian_reg=0, init_step_length=1.0, softmax_reg=None):
        super().__init__()
        self.score_predictor = score_predictor
        self.num_iter = num_iter
        self.compute_losses = compute_losses
        self.detach_length = detach_length
        self.steplength_reg = steplength_reg
        self.hessian_reg = hessian_reg
        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.softmax_reg = softmax_reg
        self._parameter_batch_dim = parameter_batch_dim

    def forward(self, meta_parameter: TensorList, num_iter=None, **kwargs):
        if not isinstance(meta_parameter, TensorList):
            meta_parameter = TensorList([meta_parameter])
        _residual_batch_dim = 1
        torch_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        num_iter = self.num_iter if num_iter is None else num_iter
        step_length_factor = torch.exp(self.log_step_length)
        label_density, sample_weight, reg_weight = self.score_predictor.init_data(meta_parameter, **kwargs)
        exp_reg = 0 if self.softmax_reg is None else math.exp(self.softmax_reg)

        def _compute_loss(scores, weights):
            num_sequences = scores.shape[_residual_batch_dim]
            return torch.sum(sample_weight.reshape(sample_weight.shape[0], -1) * (torch.log(scores.exp().sum(dim=(-2, -1)) + exp_reg) - (label_density * scores).sum(dim=(-2, -1)))) / num_sequences + reg_weight * sum((weights * weights).sum()) / num_sequences
        meta_parameter_iterates = [meta_parameter]
        losses = []
        for i in range(num_iter):
            if i > 0 and i % self.detach_length == 0:
                meta_parameter = meta_parameter.detach()
            meta_parameter.requires_grad_(True)
            scores = self.score_predictor(meta_parameter, **kwargs)
            if self.compute_losses:
                losses.append(_compute_loss(scores, meta_parameter))
            scores_softmax = activation.softmax_reg(scores.reshape(*scores.shape[:2], -1), dim=2, reg=self.softmax_reg).reshape(scores.shape)
            dLds = sample_weight * (scores_softmax - label_density)
            weights_grad = TensorList(torch.autograd.grad(scores, meta_parameter, dLds, create_graph=True)) + meta_parameter * reg_weight
            scores_grad = torch.autograd.grad(weights_grad, dLds, weights_grad, create_graph=True)[0]
            sm_scores_grad = scores_softmax * scores_grad
            hes_scores_grad = sm_scores_grad - scores_softmax * torch.sum(sm_scores_grad, dim=(-2, -1), keepdim=True) + self.hessian_reg * scores_grad
            grad_hes_grad = (scores_grad * hes_scores_grad).reshape(*scores.shape[:2], -1).sum(dim=2).clamp(min=0)
            grad_hes_grad = (sample_weight.reshape(sample_weight.shape[0], -1) * grad_hes_grad).sum(dim=0)
            gg = (weights_grad * weights_grad).reshape(scores.shape[1], -1).sum(dim=1)
            alpha_num = sum(gg)
            alpha_den = (grad_hes_grad + sum(gg * reg_weight) + self.steplength_reg * alpha_num).clamp(1e-08)
            alpha = step_length_factor * (alpha_num / alpha_den)
            step = weights_grad.apply(lambda e: alpha.reshape([(-1 if d == self._parameter_batch_dim else 1) for d in range(e.dim())]) * e)
            meta_parameter = meta_parameter - step
            meta_parameter_iterates.append(meta_parameter)
        if self.compute_losses:
            losses.append(_compute_loss(self.score_predictor(meta_parameter, **kwargs), meta_parameter))
        torch.set_grad_enabled(torch_grad_enabled)
        if not torch_grad_enabled:
            meta_parameter.detach_()
            for w in meta_parameter_iterates:
                w.detach_()
            for l in losses:
                l.detach_()
        return meta_parameter, meta_parameter_iterates, losses


class RTSDecoder(nn.Module):
    """ Decoder module """

    def __init__(self, in_channels=1, out_channels=32, ft_channels=None, use_bn=False):
        super().__init__()
        assert ft_channels is not None
        self.ft_channels = ft_channels
        self.TSE = nn.ModuleDict()
        self.RRB1 = nn.ModuleDict()
        self.CAB = nn.ModuleDict()
        self.RRB2 = nn.ModuleDict()
        self.proj = nn.ModuleDict()
        ic = in_channels
        oc = {'layer1': 1, 'layer2': 2, 'layer3': 2, 'layer4': 4}
        out_feature_channels = {}
        if 'layer4' in ft_channels.keys():
            last_layer = 'layer4'
        else:
            last_layer = 'layer3'
        prev_layer = None
        for L, fc in self.ft_channels.items():
            if not L == last_layer:
                self.proj[L] = nn.Sequential(conv(oc[prev_layer] * out_channels, oc[L] * out_channels, 1), relu())
            self.TSE[L] = TSE(fc, ic, oc[L] * out_channels)
            self.RRB1[L] = RRB(oc[L] * out_channels, use_bn=use_bn)
            self.CAB[L] = CAB(oc[L] * out_channels, L == last_layer)
            self.RRB2[L] = RRB(oc[L] * out_channels, use_bn=use_bn)
            out_feature_channels['{}_dec'.format(L)] = oc[L] * out_channels
            prev_layer = L
        self.project = Upsampler(out_channels)
        self._out_feature_channels = out_feature_channels

    def out_feature_channels(self):
        return self._out_feature_channels

    def forward(self, scores, features, image_size, output_layers=(), num_objects=None):
        if num_objects is None:
            assert scores.dim() == 5
        else:
            assert scores.dim() == 6
        outputs = OrderedDict()
        scores = scores.view(-1, *scores.shape[-3:])
        x = None
        for i, L in enumerate(self.ft_channels):
            ft = features[L]
            s = interpolate(scores, ft.shape[-2:])
            if not x is None:
                x = self.proj[L](x)
            if num_objects is not None:
                h, hpool = self.TSE[L](ft.view(ft.shape[0], 1, *ft.shape[-3:]).repeat(1, num_objects, 1, 1, 1).view(-1, *ft.shape[-3:]), s, x)
            else:
                h, hpool = self.TSE[L](ft, s, x)
            h = self.RRB1[L](h)
            h = self.CAB[L](hpool, h)
            x = self.RRB2[L](h)
            if '{}_dec'.format(L) in output_layers:
                outputs['{}_dec'.format(L)] = x
        x = self.project(x, image_size)
        return x, outputs


class ResidualDS16SW_Clf(nn.Module):
    """ Outputs the few-shot learner label and spatial importance weights given the segmentation mask """

    def __init__(self, layer_dims, use_bn=True):
        super().__init__()
        self.conv_block = conv_block(1, layer_dims[0], kernel_size=3, stride=1, padding=1, batch_norm=use_bn)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=1)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=1, downsample=ds1, use_bn=use_bn)
        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=1)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=1, downsample=ds2, use_bn=use_bn)
        self.label_pred = conv_block(layer_dims[2], layer_dims[3], kernel_size=3, stride=1, padding=1, relu=True, batch_norm=use_bn)
        self.samp_w_pred = nn.Conv2d(layer_dims[2], layer_dims[3], kernel_size=3, padding=1, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.samp_w_pred.weight.data.fill_(0)
        self.samp_w_pred.bias.data.fill_(1)

    def forward(self, label_mask, feature=None):
        assert label_mask.dim() == 4
        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])
        out = self.pool(self.conv_block(label_mask))
        out = self.res2(self.res1(out))
        label_enc = self.label_pred(out)
        sample_w = self.samp_w_pred(out)
        label_enc = label_enc.view(label_shape[0], label_shape[1], *label_enc.shape[-3:])
        sample_w = sample_w.view(label_shape[0], label_shape[1], *sample_w.shape[-3:])
        return label_enc, sample_w


class LearnersFusion(nn.Module):
    """  """

    def __init__(self, fusion_type):
        super().__init__()
        self.fusion_type = fusion_type
        if self.fusion_type == 'concat':
            self.fusion_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, seg_learner_out, clf_learner_out):
        assert seg_learner_out.shape == clf_learner_out.shape
        assert seg_learner_out.shape[0] == 1
        if self.fusion_type == 'add':
            return seg_learner_out + clf_learner_out
        if self.fusion_type == 'concat':
            concat_output = torch.cat([seg_learner_out, clf_learner_out], dim=2)
            concat_output = concat_output.squeeze(0)
            concat_output = self.fusion_conv1(concat_output)
            concat_output = concat_output.unsqueeze(0)
            return concat_output
        None
        assert False


class RTSResidual(nn.Module):
    """ Computes the residuals W(y_t)*(T_tau(x_t) - E(y_t) and lambda*tau in the few-shot learner loss (3) in the
    paper """

    def __init__(self, init_filter_reg=0.01, filter_dilation_factors=None):
        super().__init__()
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.filter_dilation_factors = filter_dilation_factors

    def forward(self, meta_parameter: TensorList, feat, label, sample_weight=None):
        filter = meta_parameter[0]
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        scores = filter_layer.apply_filter(feat, filter, dilation_factors=self.filter_dilation_factors)
        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images)
        elif isinstance(sample_weight, torch.Tensor):
            if sample_weight.numel() == scores.numel():
                sample_weight = sample_weight.view(scores.shape)
            elif sample_weight.dim() == 1:
                sample_weight = sample_weight.view(-1, 1, 1, 1, 1)
        label = label.view(scores.shape)
        data_residual = sample_weight * (scores - label)
        reg_residual = self.filter_reg * filter.view(1, num_sequences, -1)
        return TensorList([data_residual, reg_residual])


class RTSNet(nn.Module):

    def __init__(self, feature_extractor, target_model, decoder, target_model_input_layer, decoder_input_layers, label_encoder=None, classifier=None, clf_encoder=None, classification_layer='layer3', clf_enc_input='baseline', box_label_encoder=None, box_label_decoder=None, box_target_model=None, box_target_model_segm=None, bbox_encoder=None, segm_encoder=None, fusion_module=None):
        super().__init__()
        self.box_target_model = box_target_model
        self.box_target_model_segm = box_target_model_segm
        self.bbox_encoder = bbox_encoder
        self.segm_encoder = segm_encoder
        self.box_label_encoder = box_label_encoder
        self.box_label_decoder = box_label_decoder
        self.target_model = target_model
        self.label_encoder = label_encoder
        self.target_model_input_layer = target_model_input_layer
        if isinstance(target_model_input_layer, str):
            self.target_model_input_layer = target_model_input_layer,
        self.clf_encoder = clf_encoder
        self.clf_enc_input = clf_enc_input
        self.classifier = classifier
        self.classification_layer = classification_layer
        if isinstance(classification_layer, str):
            self.classification_layer = classification_layer,
        self.feature_extractor = feature_extractor
        self.decoder = decoder
        self.fusion_module = fusion_module
        self.decoder_input_layers = decoder_input_layers
        self.output_layers = sorted(list(set(self.target_model_input_layer + self.decoder_input_layers)))

    def forward_box_mask_sta(self, train_imgs, train_bb):
        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]
        train_feat_backbone = self.extract_backbone_features(train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        train_feat_tm = self.extract_target_model_features(train_feat_backbone)
        train_bbox_enc, _ = self.box_label_encoder(train_bb, train_feat_tm, list(train_imgs.shape[-2:]))
        train_mask_enc, train_mask_sw = self.bbox_encoder(train_bb, train_feat_tm, list(train_imgs.shape[-2:]))
        train_feat_tm = train_feat_tm.view(num_train_frames, num_sequences, *train_feat_tm.shape[-3:])
        _, filter_iter, _ = self.box_target_model.get_filter(train_feat_tm, train_mask_enc, train_mask_sw)
        target_scores = [self.box_target_model.apply_target_model(f, train_feat_tm) for f in filter_iter]
        target_scores_last_iter = target_scores[-1]
        coarse_mask = torch.cat((train_bb, target_scores_last_iter), dim=2)
        pred_all, _ = self.box_label_decoder(coarse_mask, train_feat_backbone, train_imgs.shape[-2:])
        pred_all = pred_all.view(num_train_frames, num_sequences, *pred_all.shape[-2:])
        train_segm_enc, train_segm_sw = self.segm_encoder(torch.sigmoid(pred_all), train_feat_tm)
        _, filter_iter_segm, _ = self.box_target_model_segm.get_filter(train_feat_tm, train_segm_enc, train_segm_sw)
        target_scores_segm = [self.box_target_model_segm.apply_target_model(f, train_feat_tm) for f in filter_iter_segm]
        target_scores_last_iter_segm = target_scores_segm[-1]
        coarse_mask = torch.cat((train_bb, target_scores_last_iter_segm), dim=2)
        pred_all_segm, _ = self.box_label_decoder(coarse_mask, train_feat_backbone, train_imgs.shape[-2:])
        pred_all_segm = pred_all_segm.view(num_train_frames, num_sequences, *pred_all_segm.shape[-2:])
        return pred_all, pred_all_segm

    def forward_classifier_only(self, train_imgs, test_imgs, train_bb, train_label):
        train_feat_backbone = self.extract_backbone_features(train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat_backbone = self.extract_backbone_features(test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))
        train_feat_clf = self.get_backbone_clf_feat(train_feat_backbone)
        test_feat_clf = self.get_backbone_clf_feat(test_feat_backbone)
        clf_target_scores = self.classifier(train_feat_clf, test_feat_clf, train_bb, train_label=train_label)
        return clf_target_scores

    def forward(self, train_imgs, test_imgs, train_masks, test_masks, train_bb, train_label, test_label, num_refinement_iter=2):
        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]
        num_test_frames = test_imgs.shape[0]
        train_feat_backbone = self.extract_backbone_features(train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat_backbone = self.extract_backbone_features(test_imgs.contiguous().view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))
        train_feat_tm = self.extract_target_model_features(train_feat_backbone)
        test_feat_tm = self.extract_target_model_features(test_feat_backbone)
        train_feat_tm = train_feat_tm.view(num_train_frames, num_sequences, *train_feat_tm.shape[-3:])
        test_feat_tm = test_feat_tm.view(num_test_frames, num_sequences, *test_feat_tm.shape[-3:])
        train_feat_tm_all = [train_feat_tm]
        train_feat_clf = self.get_backbone_clf_feat(train_feat_backbone)
        test_feat_clf = self.get_backbone_clf_feat(test_feat_backbone)
        clf_target_scores = self.classifier(train_feat_clf, test_feat_clf, train_bb, train_label=train_label)
        few_shot_label, few_shot_sw = self.label_encoder(train_masks, train_feat_tm)
        few_shot_label_all = [few_shot_label]
        few_shot_sw_all = None if few_shot_sw is None else [few_shot_sw]
        if self.clf_enc_input in ['baseline', 'gt']:
            clf_input = test_label
        elif self.clf_enc_input == 'sc':
            clf_input = clf_target_scores[-1]
        else:
            None
            assert False
        encoded_bbox_labels, _ = self.clf_encoder(clf_input)
        filter, filter_iter, _ = self.target_model.get_filter(train_feat_tm, few_shot_label, few_shot_sw)
        mask_predictons_all = []
        for i in range(num_test_frames):
            test_feat_tm_it = test_feat_tm.view(num_test_frames, num_sequences, *test_feat_tm.shape[-3:])[i:i + 1, ...]
            mask_encoding_pred = [self.target_model.apply_target_model(f, test_feat_tm_it) for f in filter_iter]
            test_feat_backbone_it = {k: v.view(num_test_frames, num_sequences, *v.shape[-3:])[i, ...] for k, v in test_feat_backbone.items()}
            mask_encoding_pred_last_iter = mask_encoding_pred[-1]
            if self.clf_enc_input == 'baseline':
                decoder_input = mask_encoding_pred_last_iter
            else:
                encoded_bbox_label = interpolate(encoded_bbox_labels[i, :, :, :, :], mask_encoding_pred_last_iter.shape[-2:])
                encoded_bbox_label = encoded_bbox_label.unsqueeze(0)
                decoder_input = self.fusion_module(mask_encoding_pred_last_iter, encoded_bbox_label)
            mask_pred, decoder_feat = self.decoder(decoder_input, test_feat_backbone_it, test_imgs.shape[-2:])
            mask_pred = mask_pred.view(1, num_sequences, *mask_pred.shape[-2:])
            mask_predictons_all.append(mask_pred)
            mask_pred_prob = torch.sigmoid(mask_pred.clone().detach())
            few_shot_label, few_shot_sw = self.label_encoder(mask_pred_prob, test_feat_tm_it)
            few_shot_label_all.append(few_shot_label)
            if few_shot_sw_all is not None:
                few_shot_sw_all.append(few_shot_sw)
            train_feat_tm_all.append(test_feat_tm_it)
            if i < num_test_frames - 1 and num_refinement_iter > 0:
                train_feat_tm_it = torch.cat(train_feat_tm_all, dim=0)
                few_shot_label_it = torch.cat(few_shot_label_all, dim=0)
                if few_shot_sw_all is not None:
                    few_shot_sw_it = torch.cat(few_shot_sw_all, dim=0)
                else:
                    few_shot_sw_it = None
                filter_updated, _, _ = self.target_model.filter_optimizer(TensorList([filter]), feat=train_feat_tm_it, label=few_shot_label_it, sample_weight=few_shot_sw_it, num_iter=num_refinement_iter)
                filter = filter_updated[0]
        mask_predictons_all = torch.cat(mask_predictons_all, dim=0)
        return mask_predictons_all, clf_target_scores

    def segment_target(self, target_filter, test_feat_tm, test_feat, encoded_clf_scores=None):
        assert target_filter.dim() == 5
        test_feat_tm = test_feat_tm.view(1, 1, *test_feat_tm.shape[-3:])
        mask_encoding_pred = self.target_model.apply_target_model(target_filter, test_feat_tm)
        decoder_input = mask_encoding_pred
        if encoded_clf_scores is not None:
            encoded_clf_scores = interpolate(encoded_clf_scores[0, :, :, :, :], mask_encoding_pred.shape[-2:])
            encoded_clf_scores = encoded_clf_scores.unsqueeze(0)
            decoder_input = self.fusion_module(mask_encoding_pred, encoded_clf_scores)
        mask_pred, decoder_feat = self.decoder(decoder_input, test_feat, (test_feat_tm.shape[-2] * 16, test_feat_tm.shape[-1] * 16))
        return mask_pred, mask_encoding_pred

    def get_backbone_target_model_features(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.target_model_input_layer})
        if len(self.target_model_input_layer) == 1:
            return feat[self.target_model_input_layer[0]]
        return feat

    def extract_target_model_features(self, backbone_feat):
        return self.target_model.extract_target_model_features(self.get_backbone_target_model_features(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))


class BaseModel(nn.Module, metaclass=ABCMeta):
    """
    What the child model is expect to declare:
        default_conf: dictionary of the default configuration of the model.
        It overwrites base_default_conf in BaseModel, and it is overwritten by
        the user-provided configuration passed to __init__.
        Configurations can be nested.
        required_data_keys: list of expected keys in the input data dictionary.
        strict_conf (optional): boolean. If false, BaseModel does not raise
        an error when the user provides an unknown configuration entry.
        _init(self, conf): initialization method, where conf is the final
        configuration object (also accessible with `self.conf`). Accessing
        unkown configuration entries will raise an error.
        _forward(self, data): method that returns a dictionary of batched
        prediction tensors based on a dictionary of batched input data tensors.
        loss(self, pred, data): method that returns a dictionary of losses,
        computed from model predictions and input data. Each loss is a batch
        of scalars, i.e. a torch.Tensor of shape (B,).
        The total loss to be optimized has the key `'total'`.
        metrics(self, pred, data): method that returns a dictionary of metrics,
        each as a batch of scalars.
    """
    base_default_conf = {'name': None, 'trainable': True, 'freeze_batch_normalization': False}
    default_conf = {}

    def __init__(self, conf=None):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        self.conf = {}
        self.conf.update(**self.base_default_conf)
        self.conf.update(**self.default_conf)
        if conf is not None:
            self.conf.update(**conf)
        self.required_data_keys = copy(self.required_data_keys)
        if not self.conf['trainable']:
            for p in self.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)

        def freeze_bn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        if self.conf['freeze_batch_normalization']:
            self.apply(freeze_bn)
        return self

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""
        for key in self.required_data_keys:
            assert key in data, 'Missing key {} in data'.format(key)
        return self._forward(data)

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError


def MLP(channels, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < n - 1:
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class KeypointEncoder(nn.Module):

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + list(layers) + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** 0.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model):
        super().__init__()
        assert d_model % h == 0
        self.dim = d_model // h
        self.h = h
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        b = query.size(0)
        query, key, value = [l(x).view(b, self.dim, self.h, -1) for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(b, self.dim * self.h, -1))


class AttentionalPropagation(nn.Module):

    def __init__(self, num_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, num_dim)
        self.mlp = MLP([num_dim * 2, num_dim * 2, num_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class GNNLayer(nn.Module):

    def __init__(self, feature_dim, layer_type):
        super().__init__()
        self.update = AttentionalPropagation(feature_dim, 4)
        assert layer_type in ['cross', 'self']
        self.type = layer_type

    def forward(self, desc0, desc1):
        if self.type == 'cross':
            src0, src1 = desc1, desc0
        elif self.type == 'self':
            src0, src1 = desc0, desc1
        else:
            raise ValueError(self.type)
        delta0, delta1 = self.update(desc0, src0), self.update(desc1, src1)
        desc0, desc1 = desc0 + delta0, desc1 + delta1
        return desc0, desc1


class AttentionalGNN(nn.Module):

    def __init__(self, feature_dim, layer_types, checkpointed=False):
        super().__init__()
        self.checkpointed = checkpointed
        self.layers = nn.ModuleList([GNNLayer(feature_dim, layer_type) for layer_type in layer_types])

    def forward(self, desc0, desc1):
        for layer in self.layers:
            if self.checkpointed:
                desc0, desc1 = torch.utils.checkpoint.checkpoint(layer, desc0, desc1, preserve_rng_state=False)
            else:
                desc0, desc1 = layer(desc0, desc1)
        return desc0, desc1


def arange_like(x, dim):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1


def log_double_softmax(scores, bin_score):
    b, m, n = scores.shape
    bin_ = bin_score[None, None, None]
    scores0 = torch.cat([scores, bin_.expand(b, m, 1)], 2)
    scores1 = torch.cat([scores, bin_.expand(b, 1, n)], 1)
    scores0 = torch.nn.functional.log_softmax(scores0, 2)
    scores1 = torch.nn.functional.log_softmax(scores1, 1)
    scores = scores.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = (scores0[:, :, :n] + scores1[:, :m, :]) / 2
    scores[:, :-1, -1] = scores0[:, :, -1]
    scores[:, -1, :-1] = scores1[:, -1, :]
    return scores


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters):
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters):
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = m * one, n * one
    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)
    couplings = torch.cat([torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1)
    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)
    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm
    return Z


def normalize_keypoints(kpts, shape_or_size):
    if isinstance(shape_or_size, (tuple, list)):
        h, w = shape_or_size[-2:]
        size = kpts.new_tensor([[w, h]])
    else:
        assert isinstance(shape_or_size, torch.Tensor)
        size = shape_or_size
    c = size / 2
    f = size.max(1, keepdim=True).values * 0.7
    return (kpts - c[:, None, :]) / f[:, None, :]


class SuperGlue(BaseModel):
    default_conf = {'input_dim': 256, 'descriptor_dim': 256, 'bottleneck_dim': None, 'weights': 'indoor', 'keypoint_encoder': [32, 64, 128, 256], 'GNN_layers': ['self', 'cross'] * 9, 'output_normalization': 'sinkhorn', 'num_sinkhorn_iterations': 50, 'filter_threshold': 0.2, 'checkpointed': False, 'loss': {'nll_weight': 1.0, 'nll_balancing': 0.5, 'reward_weight': 0.0, 'bottleneck_l2_weight': 0.0}}
    required_data_keys = ['img_coords0', 'img_coords1', 'descriptors0', 'descriptors1', 'scores0', 'scores1']

    def __init__(self, conf=None):
        super().__init__(conf=conf)
        if self.conf['bottleneck_dim'] is not None:
            self.bottleneck_down = nn.Conv1d(self.conf['input_dim'], self.conf['bottleneck_dim'], kernel_size=1, bias=True)
            self.bottleneck_up = nn.Conv1d(self.conf['bottleneck_dim'], self.conf['input_dim'], kernel_size=1, bias=True)
            nn.init.constant_(self.bottleneck_down.bias, 0.0)
            nn.init.constant_(self.bottleneck_up.bias, 0.0)
        if self.conf['input_dim'] != self.conf['descriptor_dim']:
            self.input_proj = nn.Conv1d(self.conf['input_dim'], self.conf['descriptor_dim'], kernel_size=1, bias=True)
            nn.init.constant_(self.input_proj.bias, 0.0)
        self.kenc = KeypointEncoder(self.conf['descriptor_dim'], self.conf['keypoint_encoder'])
        if not self.conf['skip_gnn']:
            self.gnn = AttentionalGNN(self.conf['descriptor_dim'], self.conf['GNN_layers'], self.conf['checkpointed'])
        self.final_proj = nn.Conv1d(self.conf['descriptor_dim'], self.conf['descriptor_dim'], kernel_size=1, bias=True)
        nn.init.constant_(self.final_proj.bias, 0.0)
        nn.init.orthogonal_(self.final_proj.weight, gain=1)
        bin_score = torch.nn.Parameter(torch.tensor(0.0))
        self.register_parameter('bin_score', bin_score)

    def _forward(self, data):
        pred = {}
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['img_coords0'], data['img_coords1']
        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {'matches0': kpts0.new_full(shape0, -1, dtype=torch.int), 'matches1': kpts1.new_full(shape1, -1, dtype=torch.int), 'match_scores0': kpts0.new_zeros(shape0), 'match_scores1': kpts1.new_zeros(shape1)}
        if self.conf['bottleneck_dim'] is not None:
            pred['down_descriptors0'] = desc0 = self.bottleneck_down(desc0)
            pred['down_descriptors1'] = desc1 = self.bottleneck_down(desc1)
            desc0 = self.bottleneck_up(desc0)
            desc1 = self.bottleneck_up(desc1)
            desc0 = nn.functional.normalize(desc0, p=2, dim=1)
            desc1 = nn.functional.normalize(desc1, p=2, dim=1)
            pred['bottleneck_descriptors0'] = desc0
            pred['bottleneck_descriptors1'] = desc1
            if self.conf['loss']['nll_weight'] == 0:
                desc0 = desc0.detach()
                desc1 = desc1.detach()
        if self.conf['input_dim'] != self.conf['descriptor_dim']:
            desc0 = self.input_proj(desc0)
            desc1 = self.input_proj(desc1)
        kpts0 = normalize_keypoints(kpts0, data['image_size0'])
        kpts1 = normalize_keypoints(kpts1, data['image_size1'])
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])
        if not self.conf['skip_gnn']:
            desc0, desc1 = self.gnn(desc0, desc1)
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.conf['descriptor_dim'] ** 0.5
        if self.conf['output_normalization'] == 'sinkhorn':
            scores = log_optimal_transport(scores, self.bin_score, iters=self.conf['num_sinkhorn_iterations'])
        elif self.conf['output_normalization'] == 'double_softmax':
            scores = log_double_softmax(scores, self.bin_score)
        else:
            raise ValueError(self.conf['output_normalization'])
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        m0, m1 = max0.indices, max1.indices
        mutual0 = arange_like(m0, 1)[None] == m1.gather(1, m0)
        mutual1 = arange_like(m1, 1)[None] == m0.gather(1, m1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
        valid0 = mutual0 & (mscores0 > self.conf['filter_threshold'])
        valid1 = mutual1 & valid0.gather(1, m1)
        m0 = torch.where(valid0, m0, m0.new_tensor(-1))
        m1 = torch.where(valid1, m1, m1.new_tensor(-1))
        return {**pred, 'log_assignment': scores, 'matches0': m0, 'matches1': m1, 'match_scores0': mscores0, 'match_scores1': mscores1, 'bin_score': self.bin_score}


class DescriptorExtractor(nn.Module):

    def __init__(self, backbone_feat_dim, descriptor_dim, kernel_size=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=backbone_feat_dim, out_channels=descriptor_dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)

    def forward(self, x, coords):
        feats = self.conv(x)
        assert torch.all(coords >= 0) and torch.all(coords < feats.shape[3])
        desc = feats[torch.arange(x.shape[0]).unsqueeze(1), :, coords[:, :, 0].long(), coords[:, :, 1].long()]
        return desc.permute(0, 2, 1)

    def get_descriptors(self, x, coords):
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)
        feats = self.conv(x)
        assert torch.all(coords >= 0) and torch.all(coords < feats.shape[3])
        desc = feats[torch.arange(x.shape[0]).unsqueeze(1), :, coords[:, :, 0].long(), coords[:, :, 1].long()]
        return desc.permute(0, 2, 1)


class TargetCandidateMatchingNetwork(nn.Module):

    def __init__(self, feature_extractor, classification_layer, descriptor_extractor, matcher):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classification_layer = classification_layer
        self.output_layers = sorted(list(set(self.classification_layer)))
        self.descriptor_extractor = descriptor_extractor
        self.matcher = matcher

    def forward(self, img_cropped0, img_cropped1, candidate_tsm_coords0, candidate_tsm_coords1, candidate_img_coords0, candidate_img_coords1, candidate_scores0, candidate_scores1, img_shape0, img_shape1, **kwargs):
        frame_feat0 = self.extract_backbone_features(img_cropped0.reshape(-1, *img_cropped0.shape[-3:]))
        frame_feat1 = self.extract_backbone_features(img_cropped1.reshape(-1, *img_cropped1.shape[-3:]))
        frame_feat_clf0 = self.get_backbone_clf_feat(frame_feat0)
        frame_feat_clf1 = self.get_backbone_clf_feat(frame_feat1)
        descriptors0 = self.descriptor_extractor(frame_feat_clf0, candidate_tsm_coords0[0])
        descriptors1 = self.descriptor_extractor(frame_feat_clf1, candidate_tsm_coords1[0])
        data = {'descriptors0': descriptors0, 'descriptors1': descriptors1, 'img_coords0': candidate_img_coords0[0], 'img_coords1': candidate_img_coords1[0], 'scores0': candidate_scores0[0], 'scores1': candidate_scores1[0], 'image_size0': img_shape0[0], 'image_size1': img_shape1[0]}
        pred = self.matcher(data)
        return pred

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat


class FilterPool(nn.Module):
    """Pool the target region in a feature map.
    args:
        filter_size:  Size of the filter.
        feature_stride:  Input feature stride.
        pool_square:  Do a square pooling instead of pooling the exact target region."""

    def __init__(self, filter_size=1, feature_stride=16, pool_square=False):
        super().__init__()
        self.prroi_pool = PrRoIPool2D(filter_size, filter_size, 1 / feature_stride)
        self.pool_square = pool_square

    def forward(self, feat, bb):
        """Pool the regions in bb.
        args:
            feat:  Input feature maps. Dims (num_samples, feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (num_samples, 4).
        returns:
            pooled_feat:  Pooled features. Dims (num_samples, feat_dim, wH, wW)."""
        bb = bb.reshape(-1, 4)
        num_images_total = bb.shape[0]
        batch_index = torch.arange(num_images_total, dtype=torch.float32).reshape(-1, 1)
        pool_bb = bb.clone()
        if self.pool_square:
            bb_sz = pool_bb[:, 2:4].prod(dim=1, keepdim=True).sqrt()
            pool_bb[:, :2] += pool_bb[:, 2:] / 2 - bb_sz / 2
            pool_bb[:, 2:] = bb_sz
        pool_bb[:, 2:4] = pool_bb[:, 0:2] + pool_bb[:, 2:4]
        roi1 = torch.cat((batch_index, pool_bb), dim=1)
        return self.prroi_pool(feat, roi1)


class FilterInitializer(nn.Module):
    """Initializes a target classification filter by applying a number of conv layers before and after pooling the target region.
    args:
        filter_size:  Size of the filter.
        feature_dim:  Input feature dimentionality.
        feature_stride:  Input feature stride.
        pool_square:  Do a square pooling instead of pooling the exact target region.
        filter_norm:  Normalize the output filter with its size in the end.
        num_filter_pre_convs:  Conv layers before pooling.
        num_filter_post_convs:  Conv layers after pooling."""

    def __init__(self, filter_size=1, feature_dim=256, feature_stride=16, pool_square=False, filter_norm=True, num_filter_pre_convs=1, num_filter_post_convs=0):
        super().__init__()
        self.filter_pool = FilterPool(filter_size=filter_size, feature_stride=feature_stride, pool_square=pool_square)
        self.filter_norm = filter_norm
        pre_conv_layers = []
        for i in range(num_filter_pre_convs):
            pre_conv_layers.append(conv_block(feature_dim, feature_dim, kernel_size=3, padding=1))
        self.filter_pre_layers = nn.Sequential(*pre_conv_layers) if pre_conv_layers else None
        post_conv_layers = []
        for i in range(num_filter_post_convs):
            post_conv_layers.append(conv_block(feature_dim, feature_dim, kernel_size=1, padding=0))
        post_conv_layers.append(nn.Conv2d(feature_dim, feature_dim, kernel_size=1, padding=0))
        self.filter_post_layers = nn.Sequential(*post_conv_layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feat, bb):
        """Runs the initializer module.
        Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
        returns:
            weights:  The output weights. Dims (sequences, feat_dim, wH, wW)."""
        num_images = bb.shape[0] if bb.dim() == 3 else 1
        if self.filter_pre_layers is not None:
            feat = self.filter_pre_layers(feat.reshape(-1, feat.shape[-3], feat.shape[-2], feat.shape[-1]))
        feat_post = self.filter_pool(feat, bb)
        weights = self.filter_post_layers(feat_post)
        if num_images > 1:
            weights = torch.mean(weights.reshape(num_images, -1, weights.shape[-3], weights.shape[-2], weights.shape[-1]), dim=0)
        if self.filter_norm:
            weights = weights / (weights.shape[1] * weights.shape[2] * weights.shape[3])
        return weights


class FilterInitializerLinear(nn.Module):
    """Initializes a target classification filter by applying a linear conv layer and then pooling the target region.
    args:
        filter_size:  Size of the filter.
        feature_dim:  Input feature dimentionality.
        feature_stride:  Input feature stride.
        pool_square:  Do a square pooling instead of pooling the exact target region.
        filter_norm:  Normalize the output filter with its size in the end.
        conv_ksz:  Kernel size of the conv layer before pooling."""

    def __init__(self, filter_size=1, feature_dim=256, feature_stride=16, pool_square=False, filter_norm=True, conv_ksz=3, init_weights='default'):
        super().__init__()
        self.filter_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=conv_ksz, padding=conv_ksz // 2)
        self.filter_pool = FilterPool(filter_size=filter_size, feature_stride=feature_stride, pool_square=pool_square)
        self.filter_norm = filter_norm
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_weights == 'default':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                elif init_weights == 'zero':
                    m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feat, bb):
        """Runs the initializer module.
        Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
        returns:
            weights:  The output weights. Dims (sequences, feat_dim, wH, wW)."""
        num_images = feat.shape[0]
        feat = self.filter_conv(feat.reshape(-1, feat.shape[-3], feat.shape[-2], feat.shape[-1]))
        weights = self.filter_pool(feat, bb)
        if num_images > 1:
            weights = torch.mean(weights.reshape(num_images, -1, weights.shape[-3], weights.shape[-2], weights.shape[-1]), dim=0)
        if self.filter_norm:
            weights = weights / (weights.shape[1] * weights.shape[2] * weights.shape[3])
        return weights


class FilterInitializerSiamese(nn.Module):
    """Initializes a target classification filter by only pooling the target region (similar to Siamese trackers).
    args:
        filter_size:  Size of the filter.
        feature_stride:  Input feature stride.
        pool_square:  Do a square pooling instead of pooling the exact target region.
        filter_norm:  Normalize the output filter with its size in the end."""

    def __init__(self, filter_size=1, feature_stride=16, pool_square=False, filter_norm=True):
        super().__init__()
        self.filter_pool = FilterPool(filter_size=filter_size, feature_stride=feature_stride, pool_square=pool_square)
        self.filter_norm = filter_norm
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feat, bb):
        """Runs the initializer module.
        Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
        returns:
            weights:  The output weights. Dims (sequences, feat_dim, wH, wW)."""
        num_images = feat.shape[0]
        feat = feat.reshape(-1, feat.shape[-3], feat.shape[-2], feat.shape[-1])
        weights = self.filter_pool(feat, bb)
        if num_images > 1:
            weights = torch.mean(weights.reshape(num_images, -1, weights.shape[-3], weights.shape[-2], weights.shape[-1]), dim=0)
        if self.filter_norm:
            weights = weights / (weights.shape[1] * weights.shape[2] * weights.shape[3])
        return weights


class DiMPSteepestDescentGN(nn.Module):
    """Optimizer module for DiMP.
    It unrolls the steepest descent with Gauss-Newton iterations to optimize the target filter.
    Moreover it learns parameters in the loss itself, as described in the DiMP paper.
    args:
        num_iter:  Number of default optimization iterations.
        feat_stride:  The stride of the input feature.
        init_step_length:  Initial scaling of the step length (which is then learned).
        init_filter_reg:  Initial filter regularization weight (which is then learned).
        init_gauss_sigma:  The standard deviation to use for the initialization of the label function.
        num_dist_bins:  Number of distance bins used for learning the loss label, mask and weight.
        bin_displacement:  The displacement of the bins (level of discritization).
        mask_init_factor:  Parameter controlling the initialization of the target mask.
        score_act:  Type of score activation (target mask computation) to use. The default 'relu' is what is described in the paper.
        act_param:  Parameter for the score_act.
        min_filter_reg:  Enforce a minimum value on the regularization (helps stability sometimes).
        mask_act:  What activation to do on the output of the mask computation ('sigmoid' or 'linear').
        detach_length:  Detach the filter every n-th iteration. Default is to never detech, i.e. 'Inf'.
        alpha_eps:  Term in the denominator of the steepest descent that stabalizes learning.
    """

    def __init__(self, num_iter=1, feat_stride=16, init_step_length=1.0, init_filter_reg=0.01, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0, mask_init_factor=4.0, score_act='relu', act_param=None, min_filter_reg=0.001, mask_act='sigmoid', detach_length=float('Inf'), alpha_eps=0):
        super().__init__()
        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)
        self.min_filter_reg = min_filter_reg
        self.detach_length = detach_length
        self.alpha_eps = alpha_eps
        d = torch.arange(num_dist_bins, dtype=torch.float32).reshape(1, -1, 1, 1) * bin_displacement
        if init_gauss_sigma == 0:
            init_gauss = torch.zeros_like(d)
            init_gauss[0, 0, 0, 0] = 1
        else:
            init_gauss = torch.exp(-1 / 2 * (d / init_gauss_sigma) ** 2)
        self.label_map_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.label_map_predictor.weight.data = init_gauss - init_gauss.min()
        mask_layers = [nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)]
        if mask_act == 'sigmoid':
            mask_layers.append(nn.Sigmoid())
            init_bias = 0.0
        elif mask_act == 'linear':
            init_bias = 0.5
        else:
            raise ValueError('Unknown activation')
        self.target_mask_predictor = nn.Sequential(*mask_layers)
        self.target_mask_predictor[0].weight.data = mask_init_factor * torch.tanh(2.0 - d) + init_bias
        self.spatial_weight_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)
        if score_act == 'bentpar':
            self.score_activation = activation.BentIdentPar(act_param)
            self.score_activation_deriv = activation.BentIdentParDeriv(act_param)
        elif score_act == 'relu':
            self.score_activation = activation.LeakyReluPar()
            self.score_activation_deriv = activation.LeakyReluParDeriv()
        else:
            raise ValueError('Unknown score activation')

    def forward(self, weights, feat, bb, sample_weight=None, num_iter=None, compute_losses=True):
        """Runs the optimizer module.
        Note that [] denotes an optional dimension.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            sample_weight:  Optional weight for each sample. Dims: (images_in_sequence, [sequences]).
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = weights.shape[-2], weights.shape[-1]
        output_sz = feat.shape[-2] + (weights.shape[-2] + 1) % 2, feat.shape[-1] + (weights.shape[-1] + 1) % 2
        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg * self.filter_reg).clamp(min=self.min_filter_reg ** 2)
        dmap_offset = torch.Tensor(filter_sz) % 2 / 2.0
        center = ((bb[..., :2] + bb[..., 2:] / 2) / self.feat_stride).reshape(-1, 2).flip((1,)) - dmap_offset
        dist_map = self.distance_map(center, output_sz)
        label_map = self.label_map_predictor(dist_map).reshape(num_images, num_sequences, *dist_map.shape[-2:])
        target_mask = self.target_mask_predictor(dist_map).reshape(num_images, num_sequences, *dist_map.shape[-2:])
        spatial_weight = self.spatial_weight_predictor(dist_map).reshape(num_images, num_sequences, *dist_map.shape[-2:])
        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images) * spatial_weight
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.sqrt().reshape(num_images, num_sequences, 1, 1) * spatial_weight
        backprop_through_learning = self.detach_length > 0
        weight_iterates = [weights]
        losses = []
        for i in range(num_iter):
            if not backprop_through_learning or i > 0 and i % self.detach_length == 0:
                weights = weights.detach()
            scores = filter_layer.apply_filter(feat, weights)
            scores_act = self.score_activation(scores, target_mask)
            score_mask = self.score_activation_deriv(scores, target_mask)
            residuals = sample_weight * (scores_act - label_map)
            if compute_losses:
                losses.append(((residuals ** 2).sum() + reg_weight * (weights ** 2).sum()) / num_sequences)
            residuals_mapped = score_mask * (sample_weight * residuals)
            weights_grad = filter_layer.apply_feat_transpose(feat, residuals_mapped, filter_sz, training=self.training) + reg_weight * weights
            scores_grad = filter_layer.apply_filter(feat, weights_grad)
            scores_grad = sample_weight * (score_mask * scores_grad)
            alpha_num = (weights_grad * weights_grad).sum(dim=(1, 2, 3))
            alpha_den = ((scores_grad * scores_grad).reshape(num_images, num_sequences, -1).sum(dim=(0, 2)) + (reg_weight + self.alpha_eps) * alpha_num).clamp(1e-08)
            alpha = alpha_num / alpha_den
            weights = weights - step_length_factor * alpha.reshape(-1, 1, 1, 1) * weights_grad
            weight_iterates.append(weights)
        if compute_losses:
            scores = filter_layer.apply_filter(feat, weights)
            scores = self.score_activation(scores, target_mask)
            losses.append((((sample_weight * (scores - label_map)) ** 2).sum() + reg_weight * (weights ** 2).sum()) / num_sequences)
        return weights, weight_iterates, losses


class DiMPL2SteepestDescentGN(nn.Module):
    """A simpler optimizer module that uses L2 loss.
    args:
        num_iter:  Number of default optimization iterations.
        feat_stride:  The stride of the input feature.
        init_step_length:  Initial scaling of the step length (which is then learned).
        gauss_sigma:  The standard deviation of the label function.
        hinge_threshold:  Threshold for the hinge-based loss (see DiMP paper).
        init_filter_reg:  Initial filter regularization weight (which is then learned).
        min_filter_reg:  Enforce a minimum value on the regularization (helps stability sometimes).
        detach_length:  Detach the filter every n-th iteration. Default is to never detech, i.e. 'Inf'.
        alpha_eps:  Term in the denominator of the steepest descent that stabalizes learning.
    """

    def __init__(self, num_iter=1, feat_stride=16, init_step_length=1.0, gauss_sigma=1.0, hinge_threshold=-999, init_filter_reg=0.01, min_filter_reg=0.001, detach_length=float('Inf'), alpha_eps=0.0):
        super().__init__()
        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.min_filter_reg = min_filter_reg
        self.detach_length = detach_length
        self.hinge_threshold = hinge_threshold
        self.gauss_sigma = gauss_sigma
        self.alpha_eps = alpha_eps

    def get_label(self, center, output_sz):
        center = center.reshape(center.shape[0], -1, center.shape[-1])
        k0 = torch.arange(output_sz[0], dtype=torch.float32).reshape(1, 1, -1, 1)
        k1 = torch.arange(output_sz[1], dtype=torch.float32).reshape(1, 1, 1, -1)
        g0 = torch.exp(-1.0 / (2 * self.gauss_sigma ** 2) * (k0 - center[:, :, 0].reshape(*center.shape[:2], 1, 1)) ** 2)
        g1 = torch.exp(-1.0 / (2 * self.gauss_sigma ** 2) * (k1 - center[:, :, 1].reshape(*center.shape[:2], 1, 1)) ** 2)
        gauss = g0 * g1
        return gauss

    def forward(self, weights, feat, bb, sample_weight=None, num_iter=None, compute_losses=True):
        """Runs the optimizer module.
        Note that [] denotes an optional dimension.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            sample_weight:  Optional weight for each sample. Dims: (images_in_sequence, [sequences]).
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = weights.shape[-2], weights.shape[-1]
        output_sz = feat.shape[-2] + (weights.shape[-2] + 1) % 2, feat.shape[-1] + (weights.shape[-1] + 1) % 2
        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg * self.filter_reg).clamp(min=self.min_filter_reg ** 2)
        dmap_offset = torch.Tensor(filter_sz) % 2 / 2.0
        center = ((bb[..., :2] + bb[..., 2:] / 2) / self.feat_stride).flip((-1,)) - dmap_offset
        label_map = self.get_label(center, output_sz)
        target_mask = (label_map > self.hinge_threshold).float()
        label_map *= target_mask
        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images)
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.sqrt().reshape(num_images, num_sequences, 1, 1)
        weight_iterates = [weights]
        losses = []
        for i in range(num_iter):
            if i > 0 and i % self.detach_length == 0:
                weights = weights.detach()
            scores = filter_layer.apply_filter(feat, weights)
            scores_act = target_mask * scores + (1.0 - target_mask) * F.relu(scores)
            score_mask = target_mask + (1.0 - target_mask) * (scores.detach() > 0).float()
            residuals = sample_weight * (scores_act - label_map)
            if compute_losses:
                losses.append(((residuals ** 2).sum() + reg_weight * (weights ** 2).sum()) / num_sequences)
            residuals_mapped = score_mask * (sample_weight * residuals)
            weights_grad = filter_layer.apply_feat_transpose(feat, residuals_mapped, filter_sz, training=self.training) + reg_weight * weights
            scores_grad = filter_layer.apply_filter(feat, weights_grad)
            scores_grad = sample_weight * (score_mask * scores_grad)
            alpha_num = (weights_grad * weights_grad).sum(dim=(1, 2, 3))
            alpha_den = ((scores_grad * scores_grad).reshape(num_images, num_sequences, -1).sum(dim=(0, 2)) + (reg_weight + self.alpha_eps) * alpha_num).clamp(1e-08)
            alpha = alpha_num / alpha_den
            weights = weights - step_length_factor * alpha.reshape(-1, 1, 1, 1) * weights_grad
            weight_iterates.append(weights)
        if compute_losses:
            scores = filter_layer.apply_filter(feat, weights)
            scores = target_mask * scores + (1.0 - target_mask) * F.relu(scores)
            losses.append((((sample_weight * (scores - label_map)) ** 2).sum() + reg_weight * (weights ** 2).sum()) / num_sequences)
        return weights, weight_iterates, losses


class PrDiMPSteepestDescentNewton(nn.Module):
    """Optimizer module for PrDiMP.
    It unrolls the steepest descent with Newton iterations to optimize the target filter. See the PrDiMP paper.
    args:
        num_iter:  Number of default optimization iterations.
        feat_stride:  The stride of the input feature.
        init_step_length:  Initial scaling of the step length (which is then learned).
        init_filter_reg:  Initial filter regularization weight (which is then learned).
        gauss_sigma:  The standard deviation to use for the label density function.
        min_filter_reg:  Enforce a minimum value on the regularization (helps stability sometimes).
        detach_length:  Detach the filter every n-th iteration. Default is to never detech, i.e. 'Inf'.
        alpha_eps:  Term in the denominator of the steepest descent that stabalizes learning.
        init_uni_weight:  Weight of uniform label distribution.
        normalize_label:  Wheter to normalize the label distribution.
        label_shrink:  How much to shrink to label distribution.
        softmax_reg:  Regularization in the denominator of the SoftMax.
        label_threshold:  Threshold probabilities smaller than this.
    """

    def __init__(self, num_iter=1, feat_stride=16, init_step_length=1.0, init_filter_reg=0.01, gauss_sigma=1.0, min_filter_reg=0.001, detach_length=float('Inf'), alpha_eps=0.0, init_uni_weight=None, normalize_label=False, label_shrink=0, softmax_reg=None, label_threshold=0.0):
        super().__init__()
        self.num_iter = num_iter
        self.feat_stride = feat_stride
        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.gauss_sigma = gauss_sigma
        self.min_filter_reg = min_filter_reg
        self.detach_length = detach_length
        self.alpha_eps = alpha_eps
        self.uni_weight = 0 if init_uni_weight is None else init_uni_weight
        self.normalize_label = normalize_label
        self.label_shrink = label_shrink
        self.softmax_reg = softmax_reg
        self.label_threshold = label_threshold

    def get_label_density(self, center, output_sz):
        center = center.reshape(center.shape[0], -1, center.shape[-1])
        k0 = torch.arange(output_sz[0], dtype=torch.float32).reshape(1, 1, -1, 1)
        k1 = torch.arange(output_sz[1], dtype=torch.float32).reshape(1, 1, 1, -1)
        dist0 = (k0 - center[:, :, 0].reshape(*center.shape[:2], 1, 1)) ** 2
        dist1 = (k1 - center[:, :, 1].reshape(*center.shape[:2], 1, 1)) ** 2
        if self.gauss_sigma == 0:
            dist0_view = dist0.reshape(-1, dist0.shape[-2])
            dist1_view = dist1.reshape(-1, dist1.shape[-1])
            one_hot0 = torch.zeros_like(dist0_view)
            one_hot1 = torch.zeros_like(dist1_view)
            one_hot0[torch.arange(one_hot0.shape[0]), dist0_view.argmin(dim=-1)] = 1.0
            one_hot1[torch.arange(one_hot1.shape[0]), dist1_view.argmin(dim=-1)] = 1.0
            gauss = one_hot0.reshape(dist0.shape) * one_hot1.reshape(dist1.shape)
        else:
            g0 = torch.exp(-1.0 / (2 * self.gauss_sigma ** 2) * dist0)
            g1 = torch.exp(-1.0 / (2 * self.gauss_sigma ** 2) * dist1)
            gauss = g0 / (2 * math.pi * self.gauss_sigma ** 2) * g1
        gauss = gauss * (gauss > self.label_threshold).float()
        if self.normalize_label:
            gauss /= gauss.sum(dim=(-2, -1), keepdim=True) + 1e-08
        label_dens = (1.0 - self.label_shrink) * ((1.0 - self.uni_weight) * gauss + self.uni_weight / (output_sz[0] * output_sz[1]))
        return label_dens

    def forward(self, weights, feat, bb, sample_weight=None, num_iter=None, compute_losses=True):
        """Runs the optimizer module.
        Note that [] denotes an optional dimension.
        args:
            weights:  Initial weights. Dims (sequences, feat_dim, wH, wW).
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image coords. Dims (images_in_sequence, [sequences], 4).
            sample_weight:  Optional weight for each sample. Dims: (images_in_sequence, [sequences]).
            num_iter:  Number of iterations to run.
            compute_losses:  Whether to compute the (train) loss in each iteration.
        returns:
            weights:  The final oprimized weights.
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""
        num_iter = self.num_iter if num_iter is None else num_iter
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = weights.shape[-2], weights.shape[-1]
        output_sz = feat.shape[-2] + (weights.shape[-2] + 1) % 2, feat.shape[-1] + (weights.shape[-1] + 1) % 2
        step_length_factor = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg * self.filter_reg).clamp(min=self.min_filter_reg ** 2)
        offset = torch.Tensor(filter_sz) % 2 / 2.0
        center = ((bb[..., :2] + bb[..., 2:] / 2) / self.feat_stride).flip((-1,)) - offset
        label_density = self.get_label_density(center, output_sz)
        if sample_weight is None:
            sample_weight = torch.Tensor([1.0 / num_images])
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.reshape(num_images, num_sequences, 1, 1)
        exp_reg = 0 if self.softmax_reg is None else math.exp(self.softmax_reg)

        def _compute_loss(scores, weights):
            return torch.sum(sample_weight.reshape(sample_weight.shape[0], -1) * (torch.log(scores.exp().sum(dim=(-2, -1)) + exp_reg) - (label_density * scores).sum(dim=(-2, -1)))) / num_sequences + reg_weight * (weights ** 2).sum() / num_sequences
        weight_iterates = [weights]
        losses = []
        for i in range(num_iter):
            if i > 0 and i % self.detach_length == 0:
                weights = weights.detach()
            scores = filter_layer.apply_filter(feat, weights)
            scores_softmax = activation.softmax_reg(scores.reshape(num_images, num_sequences, -1), dim=2, reg=self.softmax_reg).reshape(scores.shape)
            res = sample_weight * (scores_softmax - label_density)
            if compute_losses:
                losses.append(_compute_loss(scores, weights))
            weights_grad = filter_layer.apply_feat_transpose(feat, res, filter_sz, training=self.training) + reg_weight * weights
            scores_grad = filter_layer.apply_filter(feat, weights_grad)
            sm_scores_grad = scores_softmax * scores_grad
            hes_scores_grad = sm_scores_grad - scores_softmax * torch.sum(sm_scores_grad, dim=(-2, -1), keepdim=True)
            grad_hes_grad = (scores_grad * hes_scores_grad).reshape(num_images, num_sequences, -1).sum(dim=2).clamp(min=0)
            grad_hes_grad = (sample_weight.reshape(sample_weight.shape[0], -1) * grad_hes_grad).sum(dim=0)
            alpha_num = (weights_grad * weights_grad).sum(dim=(1, 2, 3))
            alpha_den = (grad_hes_grad + (reg_weight + self.alpha_eps) * alpha_num).clamp(1e-08)
            alpha = alpha_num / alpha_den
            weights = weights - step_length_factor * alpha.reshape(-1, 1, 1, 1) * weights_grad
            weight_iterates.append(weights)
        if compute_losses:
            scores = filter_layer.apply_filter(feat, weights)
            losses.append(_compute_loss(scores, weights))
        return weights, weight_iterates, losses


class LinearFilterLearnGen(nn.Module):

    def __init__(self, feat_stride=16, init_filter_reg=0.01, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0, mask_init_factor=4.0, score_act='bentpar', act_param=None, mask_act='sigmoid'):
        super().__init__()
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.feat_stride = feat_stride
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)
        d = torch.arange(num_dist_bins, dtype=torch.float32).reshape(1, -1, 1, 1) * bin_displacement
        if init_gauss_sigma == 0:
            init_gauss = torch.zeros_like(d)
            init_gauss[0, 0, 0, 0] = 1
        else:
            init_gauss = torch.exp(-1 / 2 * (d / init_gauss_sigma) ** 2)
        self.label_map_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.label_map_predictor.weight.data = init_gauss - init_gauss.min()
        mask_layers = [nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)]
        if mask_act == 'sigmoid':
            mask_layers.append(nn.Sigmoid())
            init_bias = 0.0
        elif mask_act == 'linear':
            init_bias = 0.5
        else:
            raise ValueError('Unknown activation')
        self.target_mask_predictor = nn.Sequential(*mask_layers)
        self.target_mask_predictor[0].weight.data = mask_init_factor * torch.tanh(2.0 - d) + init_bias
        self.spatial_weight_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)
        if score_act == 'bentpar':
            self.score_activation = activation.BentIdentPar(act_param)
        elif score_act == 'relu':
            self.score_activation = activation.LeakyReluPar()
        else:
            raise ValueError('Unknown activation')

    def forward(self, meta_parameter: TensorList, feat, bb, sample_weight=None, is_distractor=None):
        filter = meta_parameter[0]
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = filter.shape[-2], filter.shape[-1]
        scores = filter_layer.apply_filter(feat, filter)
        center = ((bb[..., :2] + bb[..., 2:] / 2) / self.feat_stride).reshape(-1, 2).flip((1,))
        if is_distractor is not None:
            center[is_distractor.reshape(-1), :] = 99999
        dist_map = self.distance_map(center, scores.shape[-2:])
        label_map = self.label_map_predictor(dist_map).reshape(num_images, num_sequences, dist_map.shape[-2], dist_map.shape[-1])
        target_mask = self.target_mask_predictor(dist_map).reshape(num_images, num_sequences, dist_map.shape[-2], dist_map.shape[-1])
        spatial_weight = self.spatial_weight_predictor(dist_map).reshape(num_images, num_sequences, dist_map.shape[-2], dist_map.shape[-1])
        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images) * spatial_weight
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.sqrt().reshape(-1, 1, 1, 1) * spatial_weight
        scores_act = self.score_activation(scores, target_mask)
        data_residual = sample_weight * (scores_act - label_map)
        reg_residual = self.filter_reg * filter.reshape(1, num_sequences, -1)
        return TensorList([data_residual, reg_residual])


class LinearFilterHinge(nn.Module):

    def __init__(self, feat_stride=16, init_filter_reg=0.01, hinge_threshold=-999, activation_leak=0.0, score_act='bentpar', act_param=None, learn_filter_reg=True):
        super().__init__()
        if learn_filter_reg:
            self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        else:
            self.filter_reg = init_filter_reg
        self.feat_stride = feat_stride
        self.hinge_threshold = hinge_threshold
        self.activation_leak = activation_leak
        if score_act == 'bentpar':
            self.score_activation = activation.BentIdentPar(act_param)
        elif score_act == 'relu':
            self.score_activation = activation.LeakyReluPar()
        else:
            raise ValueError('Unknown activation')

    def forward(self, meta_parameter: TensorList, feat, bb=None, train_label=None, sample_weight=None, is_distractor=None):
        assert isinstance(meta_parameter, TensorList)
        filter = meta_parameter[0]
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        scores = filter_layer.apply_filter(feat, filter)
        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images)
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.sqrt()
        else:
            raise NotImplementedError()
        target_mask = ((train_label > self.hinge_threshold).float() + self.activation_leak).clamp(max=1.0)
        scores_act = self.score_activation(scores, target_mask)
        data_residual = sample_weight * (scores_act - target_mask * train_label)
        reg_residual = self.filter_reg * filter.view(1, num_sequences, -1)
        return TensorList([data_residual, reg_residual])


class DiMPnet(nn.Module):
    """The DiMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression."""

    def __init__(self, feature_extractor, classifier, bb_regressor, classification_layer, bb_regressor_layer):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.classification_layer = [classification_layer] if isinstance(classification_layer, str) else classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        self.output_layers = sorted(list(set(self.classification_layer + self.bb_regressor_layer)))

    def forward(self, train_imgs, test_imgs, train_bb, test_proposals, *args, **kwargs):
        """Runs the DiMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            test_proposals:  Proposal boxes to use for the IoUNet (bb_regressor) module.
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            iou_pred:  Predicted IoU scores for the test_proposals."""
        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))
        train_feat_clf = self.get_backbone_clf_feat(train_feat)
        test_feat_clf = self.get_backbone_clf_feat(test_feat)
        target_scores = self.classifier(train_feat_clf, test_feat_clf, train_bb, *args, **kwargs)
        train_feat_iou = self.get_backbone_bbreg_feat(train_feat)
        test_feat_iou = self.get_backbone_bbreg_feat(test_feat)
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou, train_bb, test_proposals)
        return target_scores, iou_pred

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer + ['classification']
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


class KYSNet(nn.Module):

    def train(self, mode=True):
        self.training = mode
        self.backbone_feature_extractor.train(False)
        self.dimp_classifier.train(False)
        self.predictor.train(mode)
        self.bb_regressor.train(mode)
        if self.motion_feat_extractor is not None:
            self.motion_feat_extractor.train(mode)
        return self

    def __init__(self, backbone_feature_extractor, dimp_classifier, predictor, bb_regressor, classification_layer, bb_regressor_layer, train_feature_extractor=True, train_iounet=True, motion_feat_extractor=None, motion_layer=()):
        super().__init__()
        assert not train_feature_extractor
        self.backbone_feature_extractor = backbone_feature_extractor
        self.dimp_classifier = dimp_classifier
        self.predictor = predictor
        self.bb_regressor = bb_regressor
        self.classification_layer = classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        self.motion_layer = list(motion_layer)
        self.output_layers = sorted(list(set([self.classification_layer] + self.bb_regressor_layer + self.motion_layer)))
        self.train_iounet = train_iounet
        self.motion_feat_extractor = motion_feat_extractor
        if not train_feature_extractor:
            for p in self.backbone_feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, test_image_cur, dimp_filters, test_label_cur, backbone_feat_prev, label_prev, anno_prev, dimp_scores_prev, state_prev, dimp_jitter_fn):
        raise NotImplementedError

    def train_classifier(self, train_imgs, train_bb):
        assert train_imgs.dim() == 5, 'Expect 5 dimensions for train'
        num_sequences = train_imgs.shape[1]
        num_train_images = train_imgs.shape[0]
        train_feat = self.extract_backbone_features(train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        train_feat_clf = train_feat[self.classification_layer]
        train_feat_clf = train_feat_clf.view(num_train_images, num_sequences, train_feat_clf.shape[-3], train_feat_clf.shape[-2], train_feat_clf.shape[-1])
        filter, train_losses = self.dimp_classifier.train_classifier(train_feat_clf, train_bb)
        return filter

    def extract_backbone_features(self, im, layers=None):
        im = im.view(-1, *im.shape[-3:])
        if layers is None:
            layers = self.output_layers
        return self.backbone_feature_extractor(im, layers)

    def get_backbone_clf_feat(self, backbone_feat):
        feat = backbone_feat[self.classification_layer]
        return feat

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_classification_feat(self, backbone_feat):
        return self.dimp_classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def get_motion_feat(self, backbone_feat):
        if self.motion_feat_extractor is not None:
            motion_feat = self.motion_feat_extractor(backbone_feat)
            return motion_feat
        else:
            return self.predictor.extract_motion_feat(backbone_feat[self.classification_layer])

    def extract_features(self, im, layers):
        if 'classification' not in layers:
            return self.backbone_feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + [self.classification_layer] if l != 'classification' and l != 'motion'])))
        all_feat = self.backbone_feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.dimp_classifier.extract_classification_feat(all_feat[self.classification_layer])
        if self.motion_feat_extractor is not None:
            motion_feat = self.motion_feat_extractor(all_feat)
            all_feat['motion'] = motion_feat
        else:
            all_feat['motion'] = self.predictor.extract_motion_feat(all_feat[self.classification_layer])
        return OrderedDict({l: all_feat[l] for l in layers})


class ToMPnet(nn.Module):
    """The ToMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        head:  Head module containing classifier and bounding box regressor.
        head_layer:  Names of the backbone layers to use for the head module."""

    def __init__(self, feature_extractor, head, head_layer):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head
        self.head_layer = [head_layer] if isinstance(head_layer, str) else head_layer
        self.output_layers = sorted(list(set(self.head_layer)))

    def forward(self, train_imgs, test_imgs, train_bb, *args, **kwargs):
        """Runs the ToMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            bbox_preds:  Predicted bounding box offsets."""
        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))
        train_feat_head = self.get_backbone_head_feat(train_feat)
        test_feat_head = self.get_backbone_head_feat(test_feat)
        test_scores, bbox_preds = self.head(train_feat_head, test_feat_head, train_bb, *args, **kwargs)
        return test_scores, bbox_preds

    def get_backbone_head_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.head_layer})
        if len(self.head_layer) == 1:
            return feat[self.head_layer[0]]
        return feat

    def extract_head_feat(self, backbone_feat):
        return self.head.extract_head_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = ['head']
        if 'head' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.head_layer if l != 'head'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_head_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


class NerfPositionalEncoding(nn.Module):

    def __init__(self, depth=10, sine_type='lin_sine', avoid_aliasing=False, max_spatial_resolution=None):
        """
        out_dim = in_dim * depth * 2
        """
        super().__init__()
        if sine_type == 'lin_sine':
            self.bases = [(i + 1) for i in range(depth)]
        elif sine_type == 'exp_sine':
            self.bases = [(2 ** i) for i in range(depth)]
        None
        if avoid_aliasing and max_spatial_resolution == None:
            raise ValueError('Please specify the maxima spatial resolution (h, w) of the feature map')
        elif avoid_aliasing:
            self.factor = max_spatial_resolution / depth
        else:
            self.factor = 1.0

    @torch.no_grad()
    def forward(self, inputs):
        out = torch.cat([torch.sin(i * self.factor * math.pi * inputs) for i in self.bases] + [torch.cos(i * self.factor * math.pi * inputs) for i in self.bases], axis=-1)
        assert torch.isnan(out).any() == False
        return out


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, sine_type='lin_sine', avoid_aliazing=False, max_spatial_resolution=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.sine = NerfPositionalEncoding(num_pos_feats // 2, sine_type, avoid_aliazing, max_spatial_resolution)

    @torch.no_grad()
    def forward(self, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-06
        y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps)
        x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps)
        pos = torch.stack([x_embed, y_embed], dim=-1)
        return self.sine(pos).permute(0, 3, 1, 2)


class FilterPredictor(nn.Module):

    def __init__(self, transformer, feature_sz, use_test_frame_encoding=True):
        super().__init__()
        self.transformer = transformer
        self.feature_sz = feature_sz
        self.use_test_frame_encoding = use_test_frame_encoding
        self.box_encoding = MLP([4, self.transformer.d_model // 4, self.transformer.d_model, self.transformer.d_model])
        self.query_embed_fg = nn.Embedding(1, self.transformer.d_model)
        if self.use_test_frame_encoding:
            self.query_embed_test = nn.Embedding(1, self.transformer.d_model)
        self.query_embed_fg_decoder = self.query_embed_fg
        self.pos_encoding = PositionEmbeddingSine(num_pos_feats=self.transformer.d_model // 2, sine_type='lin_sine', avoid_aliazing=True, max_spatial_resolution=feature_sz)

    def forward(self, train_feat, test_feat, train_label, train_ltrb_target, *args, **kwargs):
        return self.predict_filter(train_feat, test_feat, train_label, train_ltrb_target, *args, **kwargs)

    def get_positional_encoding(self, feat):
        nframes, nseq, _, h, w = feat.shape
        mask = torch.zeros((nframes * nseq, h, w), dtype=torch.bool, device=feat.device)
        pos = self.pos_encoding(mask)
        return pos.reshape(nframes, nseq, -1, h, w)

    def predict_filter(self, train_feat, test_feat, train_label, train_ltrb_target, *args, **kwargs):
        if train_feat.dim() == 4:
            train_feat = train_feat.unsqueeze(1)
        if test_feat.dim() == 4:
            test_feat = test_feat.unsqueeze(1)
        if train_ltrb_target.dim() == 4:
            train_ltrb_target = train_ltrb_target.unsqueeze(1)
        h, w = test_feat.shape[-2:]
        test_pos = self.get_positional_encoding(test_feat)
        train_pos = self.get_positional_encoding(train_feat)
        test_feat_seq = test_feat.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)
        train_feat_seq = train_feat.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)
        train_label_seq = train_label.permute(1, 0, 2, 3).flatten(1).permute(1, 0).unsqueeze(2)
        train_ltrb_target_seq_T = train_ltrb_target.permute(1, 2, 0, 3, 4).flatten(2)
        test_pos = test_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)
        train_pos = train_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)
        fg_token = self.query_embed_fg.weight.reshape(1, 1, -1)
        train_label_enc = fg_token * train_label_seq
        train_ltrb_target_enc = self.box_encoding(train_ltrb_target_seq_T).permute(2, 0, 1)
        if self.use_test_frame_encoding:
            test_token = self.query_embed_test.weight.reshape(1, 1, -1)
            test_label_enc = torch.ones_like(test_feat_seq) * test_token
            feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq + test_label_enc], dim=0)
        else:
            feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq], dim=0)
        pos = torch.cat([train_pos, test_pos], dim=0)
        output_embed, enc_mem = self.transformer(feat, mask=None, query_embed=self.query_embed_fg_decoder.weight, pos_embed=pos)
        enc_opt = enc_mem[-h * w:].transpose(0, 1)
        dec_opt = output_embed.squeeze(0).transpose(1, 2)
        return dec_opt.reshape(test_feat.shape[1], -1, 1, 1), enc_opt.permute(0, 2, 1).reshape(test_feat.shape)

    def predict_cls_bbreg_filters_parallel(self, train_feat, test_feat, train_label, num_gth_frames, train_ltrb_target, *args, **kwargs):
        if train_feat.dim() == 4:
            train_feat = train_feat.unsqueeze(1)
        if test_feat.dim() == 4:
            test_feat = test_feat.unsqueeze(1)
        if train_ltrb_target.dim() == 4:
            train_ltrb_target = train_ltrb_target.unsqueeze(1)
        h, w = test_feat.shape[-2:]
        H, W = train_feat.shape[-2:]
        train_feat_stack = torch.cat([train_feat, train_feat], dim=1)
        test_feat_stack = torch.cat([test_feat, test_feat], dim=1)
        train_label_stack = torch.cat([train_label, train_label], dim=1)
        train_ltrb_target_stack = torch.cat([train_ltrb_target, train_ltrb_target], dim=1)
        test_pos = self.get_positional_encoding(test_feat)
        train_pos = self.get_positional_encoding(train_feat)
        test_feat_seq = test_feat_stack.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)
        train_feat_seq = train_feat_stack.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)
        train_label_seq = train_label_stack.permute(1, 0, 2, 3).flatten(1).permute(1, 0).unsqueeze(2)
        train_ltrb_target_seq_T = train_ltrb_target_stack.permute(1, 2, 0, 3, 4).flatten(2)
        test_pos = test_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)
        train_pos = train_pos.permute(1, 2, 0, 3, 4).flatten(2).permute(2, 0, 1)
        fg_token = self.query_embed_fg.weight.reshape(1, 1, -1)
        train_label_enc = fg_token * train_label_seq
        train_ltrb_target_enc = self.box_encoding(train_ltrb_target_seq_T).permute(2, 0, 1)
        if self.use_test_frame_encoding:
            test_token = self.query_embed_test.weight.reshape(1, 1, -1)
            test_label_enc = torch.ones_like(test_feat_seq) * test_token
            feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq + test_label_enc], dim=0)
        else:
            feat = torch.cat([train_feat_seq + train_label_enc + train_ltrb_target_enc, test_feat_seq], dim=0)
        pos = torch.cat([train_pos, test_pos], dim=0)
        src_key_padding_mask = torch.zeros(feat.shape[1], feat.shape[0]).bool()
        src_key_padding_mask[1, num_gth_frames * H * W:-h * w] = 1.0
        src_key_padding_mask = src_key_padding_mask.bool()
        output_embed, enc_mem = self.transformer(feat, mask=src_key_padding_mask, query_embed=self.query_embed_fg_decoder.weight, pos_embed=pos)
        enc_opt = enc_mem[-h * w:].transpose(0, 1).permute(0, 2, 1).reshape(test_feat_stack.shape)
        dec_opt = output_embed.squeeze(0).transpose(1, 2).reshape(test_feat_stack.shape[1], -1, 1, 1)
        cls_enc_opt = enc_opt[:, 0].unsqueeze(1)
        bbreg_enc_opt = enc_opt[:, 1].unsqueeze(1)
        cls_dec_opt = dec_opt[0].unsqueeze(0)
        bbreg_dec_opt = dec_opt[1].unsqueeze(0)
        return cls_dec_opt, bbreg_dec_opt, cls_enc_opt, bbreg_enc_opt


class Head(nn.Module):
    """
    """

    def __init__(self, filter_predictor, feature_extractor, classifier, bb_regressor, separate_filters_for_cls_and_bbreg=False):
        super().__init__()
        self.filter_predictor = filter_predictor
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.separate_filters_for_cls_and_bbreg = separate_filters_for_cls_and_bbreg

    def forward(self, train_feat, test_feat, train_bb, *args, **kwargs):
        assert train_bb.dim() == 3
        num_sequences = train_bb.shape[1]
        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])
        train_feat = self.extract_head_feat(train_feat, num_sequences)
        test_feat = self.extract_head_feat(test_feat, num_sequences)
        cls_filter, breg_filter, test_feat_enc = self.get_filter_and_features(train_feat, test_feat, *args, **kwargs)
        target_scores = self.classifier(test_feat_enc, cls_filter)
        bbox_preds = self.bb_regressor(test_feat_enc, breg_filter)
        return target_scores, bbox_preds

    def extract_head_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)
        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def get_filter_and_features(self, train_feat, test_feat, train_label, *args, **kwargs):
        if self.separate_filters_for_cls_and_bbreg:
            cls_weights, bbreg_weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
        else:
            weights, test_feat_enc = self.filter_predictor(train_feat, test_feat, train_label, *args, **kwargs)
            cls_weights = bbreg_weights = weights
        return cls_weights, bbreg_weights, test_feat_enc

    def get_filter_and_features_in_parallel(self, train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs):
        cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc = self.filter_predictor.predict_cls_bbreg_filters_parallel(train_feat, test_feat, train_label, num_gth_frames, *args, **kwargs)
        return cls_weights, bbreg_weights, cls_test_feat_enc, bbreg_test_feat_enc


class LinearFilterClassifier(nn.Module):

    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter
        if project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)

    def forward(self, feat, filter):
        if self.project_filter:
            filter_proj = self.linear(filter.reshape(-1, self.num_channels)).reshape(filter.shape)
        else:
            filter_proj = filter
        return filter_layer.apply_filter(feat, filter_proj)


def conv_layer(inplanes, outplanes, kernel_size=3, stride=1, padding=1, dilation=1):
    layers = [nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation), nn.GroupNorm(1, outplanes), nn.ReLU(inplace=True)]
    return layers


class DenseBoxRegressor(nn.Module):

    def __init__(self, num_channels, project_filter=True):
        super().__init__()
        self.num_channels = num_channels
        self.project_filter = project_filter
        if self.project_filter:
            self.linear = nn.Linear(self.num_channels, self.num_channels)
        layers = []
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        layers.extend(conv_layer(num_channels, num_channels))
        self.tower = nn.Sequential(*layers)
        self.bbreg_layer = nn.Conv2d(num_channels, 4, kernel_size=3, dilation=1, padding=1)

    def forward(self, feat, filter):
        nf, ns, c, h, w = feat.shape
        if self.project_filter:
            filter_proj = self.linear(filter.reshape(ns, c)).reshape(ns, c, 1, 1)
        else:
            filter_proj = filter
        attention = filter_layer.apply_filter(feat, filter_proj)
        feats_att = attention.unsqueeze(2) * feat
        feats_tower = self.tower(feats_att.reshape(-1, self.num_channels, feat.shape[-2], feat.shape[-1]))
        ltrb = torch.exp(self.bbreg_layer(feats_tower)).unsqueeze(0)
        return ltrb


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, pos=pos, query_pos=query_pos, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu/glu, not {activation}.')


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoderInstance(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, return_intermediate_dec=False):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        if query_embed.dim() == 2:
            query_embed = query_embed.unsqueeze(1).repeat(1, src.shape[1], 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, src, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoderInstance(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, return_intermediate_dec=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        query_embed = query_embed.unsqueeze(1).repeat(1, src.shape[1], 1)
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionalPropagation,
     lambda: ([], {'num_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BentIdentPar,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BentIdentParDeriv,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (FilterInitializerZero,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GNNLayer,
     lambda: ([], {'feature_dim': 4, 'layer_type': 'cross'}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (InstanceL2Norm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InterpCat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (IsTargetCellLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (KLRegression,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (KLRegressionGrid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (LBHinge,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LBHingev2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LeakyReluPar,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LeakyReluParDeriv,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'input_sz': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LovaszHingeWithLogitsLoss,
     lambda: ([], {'per_image': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLU,
     lambda: ([], {'min_val': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MobileBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernal_size': 4, 'stride': 1, 'nonLinear': 4, 'SE': 4, 'exp_size': 4}),
     lambda: ([torch.rand([4, 4, 2, 2])], {}),
     True),
    (MobileNetV3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (MultiHeadedAttention,
     lambda: ([], {'h': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (NerfPositionalEncoding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (RRB,
     lambda: ([], {'oc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualDS16SW,
     lambda: ([], {'layer_dims': [4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResidualDS16SW_Clf,
     lambda: ([], {'layer_dims': [4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SpatialCrossMapLRN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SqueezeBlock,
     lambda: ([], {'exp_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StableBCELoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (TSE,
     lambda: ([], {'fc': 4, 'ic': 4, 'oc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (TrackingClassificationAccuracy,
     lambda: ([], {'threshold': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (TransformerDecoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (TransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (h_sigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (h_swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_visionml_pytracking(_paritybench_base):
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

