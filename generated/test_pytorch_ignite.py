import sys
_module = sys.modules[__name__]
del sys
trigger_circle_ci = _module
teaser = _module
test_image = _module
conf = _module
main = _module
utils = _module
benchmark_fp32 = _module
benchmark_nvidia_apex = _module
benchmark_torch_cuda_amp = _module
utils = _module
main = _module
pact = _module
utils = _module
mnist_with_clearml_logger = _module
mnist_with_neptune_logger = _module
mnist_with_tensorboard_logger = _module
mnist_with_tqdm_logger = _module
mnist_with_visdom_logger = _module
mnist_with_wandb_logger = _module
dataset = _module
main = _module
model = _module
utils = _module
handlers = _module
neural_style = _module
transformer_net = _module
vgg = _module
dcgan = _module
mnist = _module
mnist_save_resume_engine = _module
mnist_with_tensorboard = _module
mnist_with_tensorboard_on_tpu = _module
mnist_with_visdom = _module
dataflow = _module
dataloaders = _module
transforms = _module
vis = _module
training = _module
exp_tracking = _module
handlers = _module
baseline_resnet50 = _module
check_baseline_resnet50 = _module
baseline_dplv3_resnet101 = _module
baseline_dplv3_resnet101_sbd = _module
eval_baseline_dplv3_resnet101_sbd = _module
dataflow = _module
main = _module
utils = _module
vis = _module
actor_critic = _module
reinforce = _module
ignite = _module
_utils = _module
base = _module
mixins = _module
contrib = _module
engines = _module
common = _module
tbptt = _module
base_logger = _module
clearml_logger = _module
lr_finder = _module
mlflow_logger = _module
neptune_logger = _module
param_scheduler = _module
polyaxon_logger = _module
tensorboard_logger = _module
time_profilers = _module
tqdm_logger = _module
visdom_logger = _module
wandb_logger = _module
metrics = _module
average_precision = _module
cohen_kappa = _module
gpu_info = _module
precision_recall_curve = _module
regression = _module
_base = _module
canberra_metric = _module
fractional_absolute_error = _module
fractional_bias = _module
geometric_mean_absolute_error = _module
geometric_mean_relative_absolute_error = _module
manhattan_distance = _module
maximum_absolute_error = _module
mean_absolute_relative_error = _module
mean_error = _module
mean_normalized_bias = _module
median_absolute_error = _module
median_absolute_percentage_error = _module
median_relative_absolute_error = _module
r2_score = _module
wave_hedges_distance = _module
roc_auc = _module
distributed = _module
auto = _module
comp_models = _module
base = _module
horovod = _module
native = _module
xla = _module
launcher = _module
utils = _module
engine = _module
deterministic = _module
engine = _module
events = _module
exceptions = _module
checkpoint = _module
early_stopping = _module
ema_handler = _module
lr_finder = _module
param_scheduler = _module
state_param_scheduler = _module
stores = _module
terminate_on_nan = _module
time_limit = _module
time_profilers = _module
timing = _module
accumulation = _module
accuracy = _module
classification_report = _module
confusion_matrix = _module
epoch_metric = _module
fbeta = _module
frequency = _module
gan = _module
fid = _module
inception_score = _module
utils = _module
loss = _module
mean_absolute_error = _module
mean_pairwise_distance = _module
mean_squared_error = _module
metric = _module
metrics_lambda = _module
multilabel_confusion_matrix = _module
nlp = _module
bleu = _module
rouge = _module
precision = _module
psnr = _module
recall = _module
root_mean_squared_error = _module
running_average = _module
ssim = _module
top_k_categorical_accuracy = _module
utils = _module
setup = _module
tests = _module
ignite = _module
test_mixins = _module
conftest = _module
test_common = _module
test_tbptt = _module
conftest = _module
test_base_logger = _module
test_clearml_logger = _module
test_mlflow_logger = _module
test_neptune_logger = _module
test_polyaxon_logger = _module
test_tensorboard_logger = _module
test_tqdm_logger = _module
test_visdom_logger = _module
test_wandb_logger = _module
test__base = _module
test_canberra_metric = _module
test_fractional_absolute_error = _module
test_fractional_bias = _module
test_geometric_mean_absolute_error = _module
test_geometric_mean_relative_absolute_error = _module
test_manhattan_distance = _module
test_maximum_absolute_error = _module
test_mean_absolute_relative_error = _module
test_mean_error = _module
test_mean_normalized_bias = _module
test_median_absolute_error = _module
test_median_absolute_percentage_error = _module
test_median_relative_absolute_error = _module
test_r2_score = _module
test_wave_hedges_distance = _module
test_average_precision = _module
test_cohen_kappa = _module
test_gpu_info = _module
test_precision_recall_curve = _module
test_roc_auc = _module
test_roc_curve = _module
check_idist_parallel = _module
test_base = _module
test_horovod = _module
test_native = _module
test_xla = _module
test_auto = _module
test_launcher = _module
utils = _module
test_horovod = _module
test_native = _module
test_serial = _module
engine = _module
test_create_supervised = _module
test_custom_events = _module
test_deterministic = _module
test_engine = _module
test_engine_state_dict = _module
test_event_handlers = _module
conftest = _module
test_checkpoint = _module
test_early_stopping = _module
test_ema_handler = _module
test_handlers = _module
test_lr_finder = _module
test_param_scheduler = _module
test_state_param_scheduler = _module
test_stores = _module
test_terminate_on_nan = _module
test_time_limit = _module
test_time_profilers = _module
test_timing = _module
test_fid = _module
test_inception_score = _module
test_utils = _module
test_bleu = _module
test_rouge = _module
test_accumulation = _module
test_accuracy = _module
test_classification_report = _module
test_confusion_matrix = _module
test_dill = _module
test_epoch_metric = _module
test_fbeta = _module
test_frequency = _module
test_loss = _module
test_mean_absolute_error = _module
test_mean_pairwise_distance = _module
test_mean_squared_error = _module
test_metric = _module
test_metrics_lambda = _module
test_multilabel_confusion_matrix = _module
test_precision = _module
test_psnr = _module
test_recall = _module
test_root_mean_squared_error = _module
test_running_average = _module
test_ssim = _module
test_top_k_categorical_accuracy = _module
test_utils = _module

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


import torch.optim as optim


from torch.optim.lr_scheduler import StepLR


from torchvision import datasets


from torchvision import models


from torchvision.transforms import Compose


from torchvision.transforms import Normalize


from torchvision.transforms import Pad


from torchvision.transforms import RandomCrop


from torchvision.transforms import RandomHorizontalFlip


from torchvision.transforms import ToTensor


from inspect import getmembers


from inspect import isclass


from inspect import isfunction


from typing import Any


from typing import Optional


from torch.cuda.amp import autocast


from torch.cuda.amp import GradScaler


from torch.nn import CrossEntropyLoss


from torch.optim import SGD


from torchvision.models import wide_resnet50_2


import random


from torch.utils.data import DataLoader


from torch.utils.data import Subset


from torchvision.datasets.cifar import CIFAR100


from torchvision.transforms import RandomErasing


import torch.nn.functional as F


from torch import nn


from torchvision.datasets import MNIST


from collections import OrderedDict


import numpy as np


from torch.optim import Adam


from torchvision import transforms


from collections import namedtuple


import warnings


import torch.utils.data as data


from torch.utils.tensorboard import SummaryWriter


from typing import Callable


from typing import Tuple


from torch.utils.data.dataset import Subset


from torchvision.datasets import ImageNet


from functools import partial


import torch.optim.lr_scheduler as lrs


from torchvision.models.resnet import resnet50


from torchvision.models.segmentation import deeplabv3_resnet101


from torch.utils.data import Dataset


from torchvision.datasets.sbd import SBDataset


from torchvision.datasets.voc import VOCSegmentation


from torch.distributions import Categorical


import numbers


from typing import cast


from typing import Dict


from typing import Iterable


from typing import Mapping


from typing import Sequence


from typing import Union


from torch.optim.optimizer import Optimizer


from torch.utils.data.distributed import DistributedSampler


import collections.abc as collections


from abc import ABCMeta


from abc import abstractmethod


from typing import List


from torch.optim import Optimizer


from collections import defaultdict


from enum import Enum


from typing import DefaultDict


from typing import Type


from typing import Iterator


from torch.utils.data import IterableDataset


from torch.utils.data.sampler import Sampler


from numbers import Number


import re


import torch.distributed as dist


import torch.multiprocessing as mp


from functools import wraps


from collections.abc import Mapping


from typing import Generator


from torch.utils.data.sampler import BatchSampler


import functools


import logging


import math


import time


from types import DynamicClassAttribute


from typing import TYPE_CHECKING


from typing import NamedTuple


from copy import deepcopy


from math import ceil


from torch.optim.lr_scheduler import _LRScheduler


import itertools


from copy import copy


from torch.optim.lr_scheduler import ReduceLROnPlateau


from typing import Collection


from torch.nn.functional import pairwise_distance


from typing import TextIO


from typing import TypeVar


from sklearn.metrics import DistanceMetric


from sklearn.metrics import r2_score


import sklearn


from sklearn.metrics import average_precision_score


from sklearn.metrics import cohen_kappa_score


from sklearn.metrics import precision_recall_curve


from sklearn.metrics import roc_auc_score


from sklearn.metrics import roc_curve


from torch.utils.data.dataloader import _InfiniteConstantSampler


from torch.utils.data.dataset import Dataset


from torch.utils.data.dataset import IterableDataset


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import SequentialSampler


from torch.utils.data.sampler import WeightedRandomSampler


from torch.nn import Linear


from torch.nn.functional import mse_loss


from torch.utils.data import BatchSampler


from torch.utils.data import RandomSampler


from torch.nn.parallel import DataParallel


from torch.nn.parallel import DistributedDataParallel


import copy


import matplotlib


from torch.optim.lr_scheduler import ExponentialLR


import scipy


from numpy import cov


import torchvision


from collections import Counter


from sklearn.metrics import accuracy_score


from sklearn.metrics import confusion_matrix


from sklearn.metrics import precision_score


from sklearn.metrics import recall_score


from sklearn.metrics import fbeta_score


from numpy.testing import assert_almost_equal


from torch.nn.functional import nll_loss


from sklearn.metrics import f1_score


from sklearn.metrics import multilabel_confusion_matrix


from sklearn.exceptions import UndefinedMetricWarning


class PACTClip(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return torch.clamp(x, 0, alpha.data)

    @staticmethod
    def backward(ctx, dy):
        x, alpha = ctx.saved_tensors
        dx = dy.clone()
        dx[x < 0] = 0
        dx[x > alpha] = 0
        dalpha = dy.clone()
        dalpha[x <= alpha] = 0
        return dx, torch.sum(dalpha)


class PACTReLU(nn.Module):

    def __init__(self, alpha=6.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        return PACTClip.apply(x, self.alpha)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, weight_bit_width=8):
    """3x3 convolution with padding"""
    return qnn.QuantConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation, weight_bit_width=weight_bit_width)


def make_PACT_relu(bit_width=8):
    relu = qnn.QuantReLU(bit_width=bit_width)
    relu.act_impl = PACTReLU()
    return relu


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, bit_width=8):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride, weight_bit_width=bit_width)
        self.bn1 = norm_layer(planes)
        self.relu = make_PACT_relu(bit_width=bit_width)
        self.conv2 = conv3x3(planes, planes, weight_bit_width=bit_width)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, stride=1, weight_bit_width=8):
    """1x1 convolution"""
    return qnn.QuantConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, weight_bit_width=weight_bit_width)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, bit_width=8):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width, weight_bit_width=bit_width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, weight_bit_width=bit_width)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, weight_bit_width=bit_width)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = make_PACT_relu(bit_width=bit_width)
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


class ResNet_QAT_Xb(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, bit_width=8):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = qnn.QuantConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = make_PACT_relu()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], bit_width=bit_width)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], bit_width=bit_width)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], bit_width=bit_width)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], bit_width=bit_width)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, bit_width=8):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride, weight_bit_width=bit_width), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, bit_width=bit_width))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, bit_width=bit_width))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class TransformerModel(nn.Module):

    def __init__(self, model_name, model_dir, dropout, n_fc, n_classes):
        super(TransformerModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=n_classes, output_hidden_states=n_fc, classifier_dropout=dropout, output_attentions=True)
        self.transformer = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=model_dir, config=self.config)

    def forward(self, inputs):
        output = self.transformer(**inputs)['logits']
        return output


class ConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class TransformerNet(torch.nn.Module):

    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class Vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class Generator(Net):
    """Generator network.

    Args:
        nf (int): Number of filters in the second-to-last deconv layer
    """

    def __init__(self, z_dim, nf, nc):
        super(Generator, self).__init__()
        self.net = nn.Sequential(nn.ConvTranspose2d(in_channels=z_dim, out_channels=nf * 8, kernel_size=4, stride=1, padding=0, bias=False), nn.BatchNorm2d(nf * 8), nn.ReLU(inplace=True), nn.ConvTranspose2d(in_channels=nf * 8, out_channels=nf * 4, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(nf * 4), nn.ReLU(inplace=True), nn.ConvTranspose2d(in_channels=nf * 4, out_channels=nf * 2, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(nf * 2), nn.ReLU(inplace=True), nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(inplace=True), nn.ConvTranspose2d(in_channels=nf, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False), nn.Tanh())
        self.weights_init()

    def forward(self, x):
        return self.net(x)


class Discriminator(Net):
    """Discriminator network.

    Args:
        nf (int): Number of filters in the first conv layer.
    """

    def __init__(self, nc, nf):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels=nc, out_channels=nf, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(nf * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=nf * 2, out_channels=nf * 4, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(nf * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=nf * 4, out_channels=nf * 8, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(nf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=nf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False), nn.Sigmoid())
        self.weights_init()

    def forward(self, x):
        output = self.net(x)
        return output.view(-1, 1).squeeze(1)


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class InceptionModel(torch.nn.Module):
    """Inception Model pre-trained on the ImageNet Dataset.

    Args:
        return_features: set it to `True` if you want the model to return features from the last pooling
            layer instead of prediction probabilities.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.
    """

    def __init__(self, return_features: bool, device: Union[str, torch.device]='cpu') ->None:
        try:
            import torchvision
            from torchvision import models
        except ImportError:
            raise ModuleNotFoundError('This module requires torchvision to be installed.')
        super(InceptionModel, self).__init__()
        self._device = device
        if Version(torchvision.__version__) < Version('0.13.0'):
            model_kwargs = {'pretrained': True}
        else:
            model_kwargs = {'weights': models.Inception_V3_Weights.DEFAULT}
        self.model = models.inception_v3(**model_kwargs)
        if return_features:
            self.model.fc = torch.nn.Identity()
        else:
            self.model.fc = torch.nn.Sequential(self.model.fc, torch.nn.Softmax(dim=1))
        self.model.eval()

    @torch.no_grad()
    def forward(self, data: torch.Tensor) ->torch.Tensor:
        if data.dim() != 4:
            raise ValueError(f'Inputs should be a tensor of dim 4, got {data.dim()}')
        if data.shape[1] != 3:
            raise ValueError(f'Inputs should be a tensor with 3 channels, got {data.shape}')
        if data.device != torch.device(self._device):
            data = data
        return self.model(data)


class DummyModel(nn.Module):

    def __init__(self, n_channels=10, out_channels=1, flatten_input=False):
        super(DummyModel, self).__init__()
        self.net = nn.Sequential(nn.Flatten() if flatten_input else nn.Identity(), nn.Linear(n_channels, out_channels))

    def forward(self, x):
        return self.net(x)


class DummyPretrainedModel(nn.Module):

    def __init__(self):
        super(DummyPretrainedModel, self).__init__()
        self.features = nn.Linear(4, 2, bias=False)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


class DummyModelMulipleParamGroups(nn.Module):

    def __init__(self):
        super(DummyModelMulipleParamGroups, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DummyPretrainedModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PACTReLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Policy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualBlock,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (UpsampleConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_pytorch_ignite(_paritybench_base):
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

