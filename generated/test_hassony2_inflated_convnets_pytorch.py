import sys
_module = sys.modules[__name__]
del sys
inflate_densenet = _module
inflate_resnet = _module
i3dense = _module
i3res = _module
inflate = _module

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


from matplotlib import pyplot as plt


import torch


import torchvision


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import math


import torch.nn.functional as F


from torch.nn import ReplicationPad3d


from torch.nn import Parameter


class _DenseLayer3d(torch.nn.Sequential):

    def __init__(self, denselayer2d, inflate_convs=False):
        super(_DenseLayer3d, self).__init__()
        self.inflate_convs = inflate_convs
        for name, child in denselayer2d.named_children():
            if isinstance(child, torch.nn.BatchNorm2d):
                self.add_module(name, inflate.inflate_batch_norm(child))
            elif isinstance(child, torch.nn.ReLU):
                self.add_module(name, child)
            elif isinstance(child, torch.nn.Conv2d):
                kernel_size = child.kernel_size[0]
                if inflate_convs and kernel_size > 1:
                    assert kernel_size % 2 == 1, 'kernel size should be                            odd be got {}'.format(kernel_size)
                    pad_size = int(kernel_size / 2)
                    pad_time = ReplicationPad3d((0, 0, 0, 0, pad_size, pad_size))
                    self.add_module('padding.1', pad_time)
                    self.add_module(name, inflate.inflate_conv(child, kernel_size))
                else:
                    self.add_module(name, inflate.inflate_conv(child, 1))
            else:
                raise ValueError('{} is not among handled layer types'.format(type(child)))
        self.drop_rate = denselayer2d.drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer3d, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _Transition3d(torch.nn.Sequential):

    def __init__(self, transition2d, inflate_conv=False):
        """
        Inflates transition layer from transition2d
        """
        super(_Transition3d, self).__init__()
        for name, layer in transition2d.named_children():
            if isinstance(layer, torch.nn.BatchNorm2d):
                self.add_module(name, inflate.inflate_batch_norm(layer))
            elif isinstance(layer, torch.nn.ReLU):
                self.add_module(name, layer)
            elif isinstance(layer, torch.nn.Conv2d):
                if inflate_conv:
                    pad_time = ReplicationPad3d((0, 0, 0, 0, 1, 1))
                    self.add_module('padding.1', pad_time)
                    self.add_module(name, inflate.inflate_conv(layer, 3))
                else:
                    self.add_module(name, inflate.inflate_conv(layer, 1))
            elif isinstance(layer, torch.nn.AvgPool2d):
                self.add_module(name, inflate.inflate_pool(layer, 2))
            else:
                raise ValueError('{} is not among handled layer types'.format(type(layer)))


def inflate_features(features, inflate_block_convs=False):
    """
    Inflates the feature extractor part of DenseNet by adding the corresponding
    inflated modules and transfering the inflated weights
    """
    features3d = torch.nn.Sequential()
    transition_nb = 0
    for name, child in features.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            features3d.add_module(name, inflate.inflate_batch_norm(child))
        elif isinstance(child, torch.nn.ReLU):
            features3d.add_module(name, child)
        elif isinstance(child, torch.nn.Conv2d):
            features3d.add_module(name, inflate.inflate_conv(child, 1))
        elif isinstance(child, torch.nn.MaxPool2d) or isinstance(child, torch.nn.AvgPool2d):
            features3d.add_module(name, inflate.inflate_pool(child))
        elif isinstance(child, torchvision.models.densenet._DenseBlock):
            block = torch.nn.Sequential()
            for nested_name, nested_child in child.named_children():
                assert isinstance(nested_child, torchvision.models.densenet._DenseLayer)
                block.add_module(nested_name, _DenseLayer3d(nested_child, inflate_convs=inflate_block_convs))
            features3d.add_module(name, block)
        elif isinstance(child, torchvision.models.densenet._Transition):
            features3d.add_module(name, _Transition3d(child))
            transition_nb = transition_nb + 1
        else:
            raise ValueError('{} is not among handled layer types'.format(type(child)))
    return features3d, transition_nb


class I3DenseNet(torch.nn.Module):

    def __init__(self, densenet2d, frame_nb, inflate_block_convs=False):
        super(I3DenseNet, self).__init__()
        self.frame_nb = frame_nb
        self.features, transition_nb = inflate_features(densenet2d.features, inflate_block_convs=inflate_block_convs)
        self.final_time_dim = frame_nb // int(math.pow(2, transition_nb))
        self.final_layer_nb = densenet2d.classifier.in_features
        self.classifier = inflate.inflate_linear(densenet2d.classifier, self.final_time_dim)

    def forward(self, inp):
        features = self.features(inp)
        out = torch.nn.functional.relu(features)
        out = torch.nn.functional.avg_pool3d(out, kernel_size=(1, 7, 7))
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(-1, self.final_time_dim * self.final_layer_nb)
        out = self.classifier(out)
        return out


def inflate_downsample(downsample2d, time_stride=1):
    downsample3d = torch.nn.Sequential(inflate.inflate_conv(downsample2d[0], time_dim=1, time_stride=time_stride, center=True), inflate.inflate_batch_norm(downsample2d[1]))
    return downsample3d


class Bottleneck3d(torch.nn.Module):

    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()
        spatial_stride = bottleneck2d.conv2.stride[0]
        self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1, center=True)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)
        self.conv2 = inflate.inflate_conv(bottleneck2d.conv2, time_dim=3, time_padding=1, time_stride=spatial_stride, center=True)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)
        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1, center=True)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)
        self.relu = torch.nn.ReLU(inplace=True)
        if bottleneck2d.downsample is not None:
            self.downsample = inflate_downsample(bottleneck2d.downsample, time_stride=spatial_stride)
        else:
            self.downsample = None
        self.stride = bottleneck2d.stride

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


def inflate_reslayer(reslayer2d):
    reslayers3d = []
    for layer2d in reslayer2d:
        layer3d = Bottleneck3d(layer2d)
        reslayers3d.append(layer3d)
    return torch.nn.Sequential(*reslayers3d)


class I3ResNet(torch.nn.Module):

    def __init__(self, resnet2d, frame_nb=16, class_nb=1000, conv_class=False):
        """
        Args:
            conv_class: Whether to use convolutional layer as classifier to
                adapt to various number of frames
        """
        super(I3ResNet, self).__init__()
        self.conv_class = conv_class
        self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=3, time_padding=1, center=True)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=3, time_padding=1, time_stride=2)
        self.layer1 = inflate_reslayer(resnet2d.layer1)
        self.layer2 = inflate_reslayer(resnet2d.layer2)
        self.layer3 = inflate_reslayer(resnet2d.layer3)
        self.layer4 = inflate_reslayer(resnet2d.layer4)
        if conv_class:
            self.avgpool = inflate.inflate_pool(resnet2d.avgpool, time_dim=1)
            self.classifier = torch.nn.Conv3d(in_channels=2048, out_channels=class_nb, kernel_size=(1, 1, 1), bias=True)
        else:
            final_time_dim = int(math.ceil(frame_nb / 16))
            self.avgpool = inflate.inflate_pool(resnet2d.avgpool, time_dim=final_time_dim)
            self.fc = inflate.inflate_linear(resnet2d.fc, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.conv_class:
            x = self.avgpool(x)
            x = self.classifier(x)
            x = x.squeeze(3)
            x = x.squeeze(3)
            x = x.mean(2)
        else:
            x = self.avgpool(x)
            x_reshape = x.view(x.size(0), -1)
            x = self.fc(x_reshape)
        return x

