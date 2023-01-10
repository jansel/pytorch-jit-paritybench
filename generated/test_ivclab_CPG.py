import sys
_module = sys.modules[__name__]
del sys
CPG_cifar100_main_normal = _module
CPG_face_main = _module
CPG_imagenet_main = _module
models = _module
layers = _module
resnet = _module
spherenet = _module
vgg = _module
packnet_cifar100_main_normal = _module
packnet_face_main = _module
packnet_imagenet_main = _module
packnet_models = _module
resnet = _module
spherenet = _module
vgg = _module
choose_appropriate_pruning_ratio_for_next_task = _module
choose_retrain_or_not = _module
random_generate_task_id = _module
LFWDataset = _module
utils = _module
cifar100_config = _module
cifar100_dataset = _module
face_dataset = _module
fine_grained_dataset = _module
manager = _module
metrics = _module
packnet_manager = _module
packnet_prune = _module
prune = _module

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


import warnings


import torch


import torch.nn as nn


import torch.optim as optim


import torch.backends.cudnn as cudnn


from torch.nn.parameter import Parameter


import logging


import math


import numpy as np


import torchvision.transforms as transforms


import torch.utils.model_zoo as model_zoo


import torch.nn.functional as F


from torch.nn.modules.utils import _pair


from torch.autograd import Variable


import torchvision.datasets as datasets


import collections


class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    @staticmethod
    def forward(ctx, inputs, threshold):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1
        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


DEFAULT_THRESHOLD = 0.005


class Ternarizer(torch.autograd.Function):
    """Ternarizes {-1, 0, 1} a real valued tensor."""

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        super(Ternarizer, self).__init__()
        self.threshold = threshold

    def forward(self, inputs):
        outputs = inputs.clone()
        outputs.fill_(0)
        outputs[inputs < 0] = -1
        outputs[inputs > self.threshold] = 1
        return outputs

    def backward(self, gradOutput):
        return gradOutput


class SharableConv2d(nn.Module):
    """Modified conv with masks for weights."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, mask_init='1s', mask_scale=0.01, threshold_fn='binarizer', threshold=None):
        super(SharableConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.mask_scale = mask_scale
        self.mask_init = mask_init
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {'threshold_fn': threshold_fn, 'threshold': threshold}
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups
        self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.piggymask = None
        if threshold_fn == 'binarizer':
            self.threshold_fn = Binarizer.apply
        elif threshold_fn == 'ternarizer':
            None
            self.threshold_fn = Ternarizer(threshold=threshold)

    def forward(self, input, layer_info=None, name=None):
        if self.piggymask is not None:
            mask_thresholded = self.threshold_fn(self.piggymask, self.info['threshold'])
            weight = mask_thresholded * self.weight
        else:
            weight = self.weight
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)
        for param in self._parameters.values():
            if param is not None:
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
        self.weight.data = fn(self.weight.data)
        if self.bias is not None and self.bias.data is not None:
            self.bias.data = fn(self.bias.data)


class SharableLinear(nn.Module):
    """Modified linear layer."""

    def __init__(self, in_features, out_features, bias=True, mask_init='1s', mask_scale=0.01, threshold_fn='binarizer', threshold=None):
        super(SharableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {'threshold_fn': threshold_fn, 'threshold': threshold}
        self.weight = Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.piggymask = None
        if threshold_fn == 'binarizer':
            self.threshold_fn = Binarizer.apply
        elif threshold_fn == 'ternarizer':
            self.threshold_fn = Ternarizer(threshold=threshold)

    def forward(self, input):
        if self.piggymask is not None:
            mask_thresholded = self.threshold_fn(self.piggymask, self.info['threshold'])
            weight = mask_thresholded * self.weight
        else:
            weight = self.weight
        return F.linear(input, weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'in_features=' + str(self.in_features) + ', out_features=' + str(self.out_features) + ')'

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)
        for param in self._parameters.values():
            if param is not None:
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)
        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)
        self.weight.data = fn(self.weight.data)
        self.bias.data = fn(self.bias.data)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, dataset_history, dataset2num_classes, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.datasets, self.classifiers = dataset_history, nn.ModuleList()
        self.dataset2num_classes = dataset2num_classes
        if self.datasets:
            self._reconstruct_classifiers()
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

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            self.classifiers.append(nn.Linear(2048, num_classes))

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def add_dataset(self, dataset, num_classes):
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            self.classifiers.append(nn.Linear(2048, num_classes))
            nn.init.normal_(self.classifiers[self.datasets.index(dataset)].weight, 0, 0.01)
            nn.init.constant_(self.classifiers[self.datasets.index(dataset)].bias, 0)

    def set_dataset(self, dataset):
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

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
        x = self.classifier(x)
        return x


class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class AngleLoss(nn.Module):

    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta, phi_theta = input
        target = target.view(-1, 1)
        index = cos_theta.data * 0.0
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = Variable(index)
        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        output = cos_theta * 1.0
        output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)
        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss.mean()
        return loss


class AngleLinear(nn.Module):

    def __init__(self, in_features, out_features, m=4):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-05).mul_(100000.0)
        self.m = m
        self.mlambda = [lambda x: x ** 0, lambda x: x ** 1, lambda x: 2 * x ** 2 - 1, lambda x: 4 * x ** 3 - 3 * x, lambda x: 8 * x ** 4 - 8 * x ** 2 + 1, lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x]

    def forward(self, input):
        x = input
        w = self.weight
        ww = w.renorm(2, 1, 1e-05).mul(100000.0)
        xlen = x.pow(2).sum(1).pow(0.5)
        wlen = ww.pow(2).sum(0).pow(0.5)
        cos_theta = x.mm(ww)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = Variable(cos_theta.data.acos())
        k = (self.m * theta / 3.14159265).floor()
        n_one = k * 0.0 - 1
        phi_theta = n_one ** k * cos_m_theta - 2 * k
        cos_theta = cos_theta * xlen.view(-1, 1)
        phi_theta = phi_theta * xlen.view(-1, 1)
        output = cos_theta, phi_theta
        return output


class SphereNet(nn.Module):

    def __init__(self, dataset_history, dataset2num_classes, network_width_multiplier=1.0, shared_layer_info={}, init_weights=True):
        super(SphereNet, self).__init__()
        self.network_width_multiplier = network_width_multiplier
        self.make_feature_layers()
        self.shared_layer_info = shared_layer_info
        self.datasets = dataset_history
        self.classifiers = nn.ModuleList()
        self.dataset2num_classes = dataset2num_classes
        if self.datasets:
            self._reconstruct_classifiers()
        if init_weights:
            self._initialize_weights()
        return

    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))
        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))
        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))
        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def forward_to_embeddings(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))
        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))
        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))
        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))
        x = self.flatten(x)
        x = self.classifier[0](x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nl.SharableConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)
        return

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            if 'face_verification' in dataset:
                embedding_size = 512
                classifier_module = nn.Sequential(nn.Linear(int(self.shared_layer_info[dataset]['network_width_multiplier'] * 512) * 7 * 7, embedding_size), AngleLinear(embedding_size, num_classes))
                self.classifiers.append(classifier_module)
            else:
                self.classifiers.append(nn.Linear(int(self.shared_layer_info[dataset]['network_width_multiplier'] * 512) * 7 * 7, num_classes))
        return

    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            if 'face_verification' in dataset:
                embedding_size = 512
                classifier_module = nn.Sequential(nn.Linear(int(self.network_width_multiplier * 512) * 7 * 7, embedding_size), AngleLinear(embedding_size, num_classes))
                self.classifiers.append(classifier_module)
                nn.init.normal_(classifier_module[0].weight, 0, 0.01)
                nn.init.constant_(classifier_module[0].bias, 0)
                nn.init.normal_(classifier_module[1].weight, 0, 0.01)
            else:
                self.classifiers.append(nn.Linear(int(self.network_width_multiplier * 512) * 7 * 7, num_classes))
                nn.init.normal_(self.classifiers[self.datasets.index(dataset)].weight, 0, 0.01)
                nn.init.constant_(self.classifiers[self.datasets.index(dataset)].bias, 0)
        return

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]
        return

    def make_feature_layers(self):
        ext = self.network_width_multiplier
        self.conv1_1 = nl.SharableConv2d(3, int(64 * ext), 3, 2, 1)
        self.relu1_1 = nn.PReLU(int(64 * ext))
        self.conv1_2 = nl.SharableConv2d(int(64 * ext), int(64 * ext), 3, 1, 1)
        self.relu1_2 = nn.PReLU(int(64 * ext))
        self.conv1_3 = nl.SharableConv2d(int(64 * ext), int(64 * ext), 3, 1, 1)
        self.relu1_3 = nn.PReLU(int(64 * ext))
        self.conv2_1 = nl.SharableConv2d(int(64 * ext), int(128 * ext), 3, 2, 1)
        self.relu2_1 = nn.PReLU(int(128 * ext))
        self.conv2_2 = nl.SharableConv2d(int(128 * ext), int(128 * ext), 3, 1, 1)
        self.relu2_2 = nn.PReLU(int(128 * ext))
        self.conv2_3 = nl.SharableConv2d(int(128 * ext), int(128 * ext), 3, 1, 1)
        self.relu2_3 = nn.PReLU(int(128 * ext))
        self.conv2_4 = nl.SharableConv2d(int(128 * ext), int(128 * ext), 3, 1, 1)
        self.relu2_4 = nn.PReLU(int(128 * ext))
        self.conv2_5 = nl.SharableConv2d(int(128 * ext), int(128 * ext), 3, 1, 1)
        self.relu2_5 = nn.PReLU(int(128 * ext))
        self.conv3_1 = nl.SharableConv2d(int(128 * ext), int(256 * ext), 3, 2, 1)
        self.relu3_1 = nn.PReLU(int(256 * ext))
        self.conv3_2 = nl.SharableConv2d(int(256 * ext), int(256 * ext), 3, 1, 1)
        self.relu3_2 = nn.PReLU(int(256 * ext))
        self.conv3_3 = nl.SharableConv2d(int(256 * ext), int(256 * ext), 3, 1, 1)
        self.relu3_3 = nn.PReLU(int(256 * ext))
        self.conv3_4 = nl.SharableConv2d(int(256 * ext), int(256 * ext), 3, 1, 1)
        self.relu3_4 = nn.PReLU(int(256 * ext))
        self.conv3_5 = nl.SharableConv2d(int(256 * ext), int(256 * ext), 3, 1, 1)
        self.relu3_5 = nn.PReLU(int(256 * ext))
        self.conv3_6 = nl.SharableConv2d(int(256 * ext), int(256 * ext), 3, 1, 1)
        self.relu3_6 = nn.PReLU(int(256 * ext))
        self.conv3_7 = nl.SharableConv2d(int(256 * ext), int(256 * ext), 3, 1, 1)
        self.relu3_7 = nn.PReLU(int(256 * ext))
        self.conv3_8 = nl.SharableConv2d(int(256 * ext), int(256 * ext), 3, 1, 1)
        self.relu3_8 = nn.PReLU(int(256 * ext))
        self.conv3_9 = nl.SharableConv2d(int(256 * ext), int(256 * ext), 3, 1, 1)
        self.relu3_9 = nn.PReLU(int(256 * ext))
        self.conv4_1 = nl.SharableConv2d(int(256 * ext), int(512 * ext), 3, 2, 1)
        self.relu4_1 = nn.PReLU(int(512 * ext))
        self.conv4_2 = nl.SharableConv2d(int(512 * ext), int(512 * ext), 3, 1, 1)
        self.relu4_2 = nn.PReLU(int(512 * ext))
        self.conv4_3 = nl.SharableConv2d(int(512 * ext), int(512 * ext), 3, 1, 1)
        self.relu4_3 = nn.PReLU(int(512 * ext))
        self.flatten = View(-1, int(ext * 512) * 7 * 7)
        return


class VGG(nn.Module):

    def __init__(self, features, dataset_history, dataset2num_classes, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.datasets, self.classifiers = dataset_history, nn.ModuleList()
        self.dataset2num_classes = dataset2num_classes
        if self.datasets:
            self._reconstruct_classifiers()
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            self.classifiers.append(nn.Linear(4096, num_classes))

    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            self.classifiers.append(nn.Linear(4096, num_classes))
            nn.init.normal_(self.classifiers[self.datasets.index(dataset)].weight, 0, 0.01)
            nn.init.constant_(self.classifiers[self.datasets.index(dataset)].bias, 0)

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]


class SphereNet20(nn.Module):

    def __init__(self, dataset_history, dataset2num_classes, shared_layer_info={}, init_weights=True):
        super(SphereNet20, self).__init__()
        self.make_feature_layers()
        self.shared_layer_info = shared_layer_info
        self.datasets = dataset_history
        self.classifiers = nn.ModuleList()
        self.dataset2num_classes = dataset2num_classes
        if self.datasets:
            self._reconstruct_classifiers()
        if init_weights != '':
            self._initialize_weights()
        return

    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))
        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))
        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))
        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def forward_to_embeddings(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))
        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))
        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))
        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))
        x = self.flatten(x)
        x = self.classifier[0](x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)
        return

    def _reconstruct_classifiers(self):
        for dataset, num_classes in self.dataset2num_classes.items():
            if 'face_verification' in dataset:
                embedding_size = 512
                classifier_module = nn.Sequential(nn.Linear(512 * 7 * 7, embedding_size), AngleLinear(embedding_size, num_classes))
                self.classifiers.append(classifier_module)
            else:
                self.classifiers.append(nn.Linear(512 * 7 * 7, num_classes))
        return

    def add_dataset(self, dataset, num_classes):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.dataset2num_classes[dataset] = num_classes
            if 'face_verification' in dataset:
                embedding_size = 512
                classifier_module = nn.Sequential(nn.Linear(512 * 7 * 7, embedding_size), AngleLinear(embedding_size, num_classes))
                self.classifiers.append(classifier_module)
                nn.init.normal_(classifier_module[0].weight, 0, 0.01)
                nn.init.constant_(classifier_module[0].bias, 0)
                nn.init.normal_(classifier_module[1].weight, 0, 0.01)
            else:
                self.classifiers.append(nn.Linear(512 * 7 * 7, num_classes))
                nn.init.normal_(self.classifiers[self.datasets.index(dataset)].weight, 0, 0.01)
                nn.init.constant_(self.classifiers[self.datasets.index(dataset)].bias, 0)

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

    def make_feature_layers(self):
        self.conv1_1 = nn.Conv2d(3, 64, 3, 2, 1)
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_3 = nn.PReLU(64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 2, 1)
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_3 = nn.PReLU(128)
        self.conv2_4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_5 = nn.PReLU(128)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 2, 1)
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_3 = nn.PReLU(256)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_5 = nn.PReLU(256)
        self.conv3_6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_7 = nn.PReLU(256)
        self.conv3_8 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_9 = nn.PReLU(256)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 2, 1)
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_3 = nn.PReLU(512)
        self.flatten = View(-1, 512 * 7 * 7)
        return


class Sequential_Debug(nn.Sequential):

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AngleLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Sequential_Debug,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SharableConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SharableLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_ivclab_CPG(_paritybench_base):
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

