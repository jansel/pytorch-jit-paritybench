import sys
_module = sys.modules[__name__]
del sys
inference = _module
train_val = _module
test = _module
videodataset = _module
vlog = _module
main = _module
backbone = _module
imagenet_pretraining = _module
basicblock = _module
bottleneck = _module
resnet = _module
resnet_based = _module
models = _module
orn_two_heads = _module
aggregation_relations = _module
classifier = _module
encoder = _module
orn = _module
two_heads = _module
rescale = _module
colormap = _module
meter = _module
other = _module
vis = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import time


import torch


import torch.nn as nn


import torch.nn.parallel


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


from torchvision.utils import save_image


import numpy as np


import torch.utils.data as data


from torchvision import transforms


import random


from torch.utils.data.dataloader import default_collate


from random import shuffle


from abc import abstractmethod


from abc import ABCMeta


import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo


import math


from torch.autograd import Variable


from torchvision import models


import numbers


from torch.nn import Module


class BasicBlock(nn.Module):
    expansion = 1
    only_2D = False

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.conv1, self.conv2 = None, None
        self.input_dim = 5
        self.dilation = dilation

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


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, dilation=(1, dilation, dilation))


class BasicBlock3D(BasicBlock):

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, **kwargs):
        super().__init__(inplanes, planes, stride, downsample, dilation)
        self.conv1 = conv3x3x3(inplanes, planes, stride, dilation)
        self.conv2 = conv3x3x3(planes, planes, dilation)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, dilation=dilation)


class BasicBlock2D(BasicBlock):

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, **kwargs):
        super().__init__(inplanes, planes, stride, downsample, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.conv2 = conv3x3(planes, planes, dilation)
        self.input_dim = 4


def conv1x3x3_conv3x1x1(in_planes, out_planes, stride=1, dilation=1, nb_temporal_conv=3):
    """3x3 convolution with padding"""
    if isinstance(stride, int):
        stride_2d, stride_1t = (1, stride, stride), (stride, 1, 1)
    else:
        stride_2d, stride_1t = (1, stride[1], stride[2]), (stride[0], 1, 1)
    _2d = nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride_2d, padding=(0, 1, 1), bias=False, dilation=dilation)
    _1t = nn.Sequential()
    for i in range(nb_temporal_conv):
        temp_conv = nn.Conv3d(out_planes, out_planes, kernel_size=(3, 1, 1), stride=stride_1t, padding=(1, 0, 0), bias=False, dilation=1)
        _1t.add_module('temp_conv_{}'.format(i), temp_conv)
        _1t.add_module('relu_{}'.format(i), nn.ReLU(inplace=True))
    return _2d, _1t


class BasicBlock2_1D(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, nb_temporal_conv=1):
        super().__init__(inplanes, planes, stride, downsample, dilation)
        self.conv1, self.conv1_1t = conv1x3x3_conv3x1x1(inplanes, planes, stride, dilation, nb_temporal_conv=nb_temporal_conv)
        self.conv2, self.conv2_1t = conv1x3x3_conv3x1x1(planes, planes, dilation, nb_temporal_conv=nb_temporal_conv)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1_1t(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.conv2_1t(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    only_2D = False

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = None
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.input_dim = 5
        self.dilation = dilation

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


class Bottleneck3D(Bottleneck):

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, **kwargs):
        super().__init__(inplanes, planes, stride, downsample, dilation)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, dilation=(1, dilation, dilation))


class Bottleneck2D(Bottleneck):

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, **kwargs):
        super().__init__(inplanes, planes, stride, downsample, dilation)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, dilation=dilation)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.input_dim = 4
        if isinstance(stride, int):
            stride_1, stride_2 = stride, stride
        else:
            stride_1, stride_2 = stride[0], stride[1]
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(stride_1, stride_2), padding=(1, 1), bias=False)


class Bottleneck2_1D(Bottleneck):

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, nb_temporal_conv=1):
        super().__init__(inplanes, planes, stride, downsample, dilation)
        if isinstance(stride, int):
            stride_2d, stride_1t = (1, stride, stride), (stride, 1, 1)
        else:
            stride_2d, stride_1t = (1, stride[1], stride[2]), (stride[0], 1, 1)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=stride_2d, padding=(0, dilation, dilation), bias=False, dilation=dilation)
        self.conv2_1t = nn.Sequential()
        for i in range(nb_temporal_conv):
            temp_conv = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=stride_1t, padding=(1, 0, 0), bias=False, dilation=1)
            self.conv2_1t.add_module('temp_conv_{}'.format(i), temp_conv)
            self.conv2_1t.add_module('relu_{}'.format(i), nn.ReLU(inplace=True))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2_1t(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


K_1st_CONV = 3


def transform_input(x, dim, T=8):
    diff = len(x.size()) - dim
    if diff > 0:
        B, C, T, W, H = x.size()
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, C, W, H)
    elif diff < 0:
        _, C, W, H = x.size()
        x = x.view(-1, T, C, W, H)
        x = x.transpose(1, 2)
    return x


class ResNet(nn.Module):

    def __init__(self, blocks, layers, num_classes=1000, str_first_conv='2D', num_final_fm=4, two_heads=False, size_fm_2nd_head=7, blocks_2nd_head=None, pooling='avg', nb_temporal_conv=1, list_stride=[1, 2, 2, 2], **kwargs):
        self.nb_temporal_conv = nb_temporal_conv
        self.size_fm_2nd_head = size_fm_2nd_head
        self.two_heads = two_heads
        self.inplanes = 64
        self.input_dim = 5
        super(ResNet, self).__init__()
        self.num_final_fm = num_final_fm
        self.time = None
        self._first_conv(str_first_conv)
        self.relu = nn.ReLU(inplace=True)
        self.list_channels = [64, 128, 256, 512]
        self.list_inplanes = []
        self.list_inplanes.append(self.inplanes)
        self.layer1 = self._make_layer(blocks[0], self.list_channels[0], layers[0], stride=list_stride[0])
        self.list_inplanes.append(self.inplanes)
        self.layer2 = self._make_layer(blocks[1], self.list_channels[1], layers[1], stride=list_stride[1])
        self.list_inplanes.append(self.inplanes)
        self.layer3 = self._make_layer(blocks[2], self.list_channels[2], layers[2], stride=list_stride[2])
        self.list_inplanes.append(self.inplanes)
        self.layer4 = self._make_layer(blocks[3], self.list_channels[3], layers[3], stride=list_stride[3])
        self.avgpool, self.avgpool_space, self.avgpool_time = None, None, None
        self.fc_classifier = nn.Linear(512 * blocks[3].expansion, num_classes)
        self.out_dim = 5
        self.pooling = pooling
        if self.two_heads:
            list_strides_2nd_head = self._get_stride_2nd_head()
            self.nb_block_common_trunk = 4 - len(blocks_2nd_head)
            self.list_layers_bis = []
            for i in range(self.nb_block_common_trunk, 4):
                self.inplanes = self.list_inplanes[i]
                layer = self._make_layer(blocks_2nd_head[i - self.nb_block_common_trunk], self.list_channels[i], layers[i], list_strides_2nd_head[i])
                if i == 0:
                    self.layer1_bis = layer
                    self.list_layers_bis.append(self.layer1_bis)
                elif i == 1:
                    self.layer2_bis = layer
                    self.list_layers_bis.append(self.layer2_bis)
                elif i == 2:
                    self.layer3_bis = layer
                    self.list_layers_bis.append(self.layer3_bis)
                elif i == 3:
                    self.layer4_bis = layer
                    self.list_layers_bis.append(self.layer4_bis)
                else:
                    raise NameError
            self.list_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        if self.pooling == 'rnn':
            cnn_features_size = 512 * blocks[3].expansion
            hidden_state_size = 256 if cnn_features_size == 512 else 512
            self.rnn = nn.GRU(input_size=cnn_features_size, hidden_size=hidden_state_size, num_layers=1, batch_first=True)
            self.fc_classifier = nn.Linear(hidden_state_size, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _get_stride_2nd_head(self):
        if self.size_fm_2nd_head == 7:
            return [1, 2, 2, 2]
        elif self.size_fm_2nd_head == 14:
            return [1, 2, 2, 1]
        elif self.size_fm_2nd_head == 28:
            return [1, 2, 1, 1]

    def _first_conv(self, str):
        self.conv1_1t = None
        self.bn1_1t = None
        if str == '3D_stabilize':
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(K_1st_CONV, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
            self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            self.bn1 = nn.BatchNorm3d(64)
        elif str == '2.5D_stabilize':
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
            self.conv1_1t = nn.Conv3d(64, 64, kernel_size=(K_1st_CONV, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
            self.bn1_1t = nn.BatchNorm3d(64)
            self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            self.bn1 = nn.BatchNorm3d(64)
        elif str == '2D':
            self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.bn1 = nn.BatchNorm2d(64)
            self.input_dim = 4
        else:
            raise NameError

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if not (block == BasicBlock2D or block == Bottleneck2D):
            stride = 1, stride, stride
        if stride != 1 or self.inplanes != planes * block.expansion:
            if block is BasicBlock2D or block is Bottleneck2D:
                conv, batchnorm = nn.Conv2d, nn.BatchNorm2d
            else:
                conv, batchnorm = nn.Conv3d, nn.BatchNorm3d
            downsample = nn.Sequential(conv(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, dilation=dilation), batchnorm(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation, nb_temporal_conv=self.nb_temporal_conv))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nb_temporal_conv=self.nb_temporal_conv))
        return nn.Sequential(*layers)

    def get_features_map(self, x, time=None, num=4, out_dim=None):
        if out_dim is None:
            out_dim = self.out_dim
        if self.time is None:
            B, C, T, W, H = x.size()
            self.time = T
        time = self.time
        x = transform_input(x, self.input_dim, T=time)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.conv1_1t is not None:
            x = self.conv1_1t(x)
            x = self.bn1_1t(x)
            x = self.relu(x)
        x = self.maxpool(x)
        if num >= 1:
            x = transform_input(x, self.layer1[0].input_dim, T=time)
            x = self.layer1(x)
        if num >= 2:
            x = transform_input(x, self.layer2[0].input_dim, T=time)
            x = self.layer2(x)
        if num >= 3:
            x = transform_input(x, self.layer3[0].input_dim, T=time)
            x = self.layer3(x)
        if num >= 4:
            x = transform_input(x, self.layer4[0].input_dim, T=time)
            x = self.layer4(x)
        return transform_input(x, out_dim, T=time)

    def get_two_heads_feature_maps(self, x, T=None, out_dim=None, heads_type='object+context'):
        x = x['clip']
        x = self.get_features_map(x, T, num=self.nb_block_common_trunk)
        if 'object' in heads_type:
            fm_objects = x
            for i in range(len(self.list_layers_bis)):
                layer = self.list_layers_bis[i]
                fm_objects = transform_input(fm_objects, layer[0].input_dim, T=T)
                fm_objects = layer(fm_objects)
            fm_objects = transform_input(fm_objects, out_dim, T=T)
        else:
            fm_objects = None
        if 'context' in heads_type:
            fm_context = x
            for i in range(self.nb_block_common_trunk, 4):
                layer = self.list_layers[i]
                fm_context = transform_input(fm_context, layer[0].input_dim, T=T)
                fm_context = layer(fm_context)
            fm_context = transform_input(fm_context, out_dim, T=T)
        else:
            fm_context = None
        return fm_context, fm_objects

    def forward(self, x):
        x = x['clip']
        x = self.get_features_map(x, num=self.num_final_fm)
        if self.pooling == 'avg':
            self.avgpool = nn.AvgPool3d((x.size(2), x.size(-1), x.size(-1))) if self.avgpool is None else self.avgpool
            x = self.avgpool(x)
        elif self.pooling == 'rnn':
            self.avgpool_space = nn.AvgPool3d((1, x.size(-1), x.size(-1))) if self.avgpool_space is None else self.avgpool_space
            x = self.avgpool_space(x)
            x = x.view(x.size(0), x.size(1), x.size(2))
            x = x.transpose(1, 2)
            ipdb.set_trace()
            x, _ = self.rnn(x)
            x = torch.mean(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc_classifier(x)
        return x


class AggregationRelations(nn.Module):

    def __init__(self):
        super(AggregationRelations, self).__init__()

    def forward(self, relational_reasoning_vector):
        B, T, K2, _ = relational_reasoning_vector.size()
        output = torch.sum(relational_reasoning_vector, 2)
        return output


class Classifier(nn.Module):

    def __init__(self, size_input, size_output):
        super(Classifier, self).__init__()
        self.size_input = size_input
        self.size_output = size_output
        self.fc = nn.Linear(self.size_input, self.size_output)

    def forward(self, x):
        preds = self.fc(x)
        return preds


class EncoderMLP(nn.Module):

    def __init__(self, input_size=149, list_hidden_size=[100, 100], relu_activation=True, p_dropout=0.5):
        super(EncoderMLP, self).__init__()
        self.input_size = input_size
        self.list_hidden_size = list_hidden_size
        self.encoder = nn.Sequential()
        current_input_size = self.input_size
        for i, hidden_size in enumerate(self.list_hidden_size):
            self.encoder.add_module('linear_{}'.format(i), nn.Linear(current_input_size, hidden_size))
            self.encoder.add_module('dropout_{}'.format(i), nn.Dropout(p=p_dropout))
            current_input_size = hidden_size
            self.encoder.add_module('relu_{}'.format(i), nn.ReLU())

    def forward(self, x):
        size_x = x.size()
        if size_x[-1] == self.input_size:
            x_input = x
        elif size_x[-1] * size_x[-2] == self.input_size:
            if len(size_x) == 5:
                B, T, K, W, H = size_x
            elif len(size_x) == 4:
                B, T, K, W = size_x
                H = 1
            x = x.contiguous()
            x_input = x.view(B, T, K, W * H)
        else:
            raise Exception
        z = self.encoder(x_input)
        return z


class ObjectRelationNetwork(nn.Module):

    def __init__(self, size_object, list_hidden_layers_size, relation_type='pairwise-inter'):
        super(ObjectRelationNetwork, self).__init__()
        self.size_object = size_object
        self.list_hidden_layers_size = list_hidden_layers_size
        self.relation_type = relation_type
        self.nb_obj = 2
        self.mlp_inter = EncoderMLP(input_size=self.nb_obj * self.size_object, list_hidden_size=self.list_hidden_layers_size)

    @staticmethod
    def create_inter_object_cat(O_1, O_2):
        list_input_mlp, input_mlp = [], None
        K = O_1.size(1)
        for k in range(K):
            O_1_k = O_2[:, (k)].unsqueeze(1).repeat(1, K, 1)
            O_1_k_input_relation = torch.cat([O_1_k, O_2], dim=2)
            list_input_mlp.append(O_1_k_input_relation)
        input_mlp = torch.cat(list_input_mlp, 1)
        return input_mlp

    @staticmethod
    def create_triwise_interactions_input(O_1, O_2):
        list_input_mlp, input_mlp = [], None
        K = O_1.size(1)
        for k1 in range(K):
            O_1_k_1 = O_2[:, (k1)].unsqueeze(1).repeat(1, K, 1)
            list_other_k = [x for x in range(K) if x != k1]
            for k2 in list_other_k:
                O_1_k_2 = O_2[:, (k2)].unsqueeze(1).repeat(1, K, 1)
                O_1_k_input_relation = torch.cat([O_1_k_1, O_1_k_2, O_2], dim=2)
                list_input_mlp.append(O_1_k_input_relation)
        input_mlp = torch.cat(list_input_mlp, 1)
        return input_mlp

    def create_input_mlp(self, O_t_1, O_t, D):
        K = O_t.size(1)
        input_mlp = self.create_inter_object_cat(O_t_1, O_t)
        is_first_obj = torch.clamp(torch.sum(input_mlp[:, :, :D], -1), 0, 1)
        is_second_obj = torch.clamp(torch.sum(input_mlp[:, :, D:], -1), 0, 1)
        is_objects = is_first_obj * is_second_obj
        return input_mlp, is_objects

    def compute_O_O_interaction(self, sets_of_objects, t, previous_T, D, sampling=False):
        O_t = sets_of_objects[:, (t)]
        list_e_inter, list_is_object_inter = [], []
        for t_1 in previous_T:
            O_t_1 = sets_of_objects[:, (t_1)]
            input_mlp_inter, is_objects_inter = self.create_input_mlp(O_t_1, O_t, D)
            e = self.mlp_inter(input_mlp_inter)
            list_e_inter.append(e)
            list_is_object_inter.append(is_objects_inter)
        if len(list_e_inter) == 1 and self.training:
            return list_e_inter[0], list_is_object_inter[0]
        else:
            all_e_inter = torch.stack(list_e_inter, 1)
            pooler = nn.AvgPool3d((all_e_inter.size(1), 1, 1))
            all_e_inter = pooler(all_e_inter)
            B, _, T_prim, D = all_e_inter.size()
            all_e_inter = all_e_inter.view(B, T_prim, D)
            is_objects_inter = torch.stack(list_is_object_inter, 1)
            is_objects_inter = torch.clamp(torch.sum(is_objects_inter, 1), 0, 1)
            return all_e_inter, is_objects_inter

    def forward(self, sets_of_objects, D, sampling=False):
        B, T, K, _ = sets_of_objects.size()
        list_e, list_is_obj = [], []
        for t in range(1, T):
            previous_T = random.sample(range(t), 1) if self.training else list(range(t))
            e_t, is_obj = self.compute_O_O_interaction(sets_of_objects, t, previous_T, D, sampling)
            list_e.append(e_t)
            list_is_obj.append(is_obj)
        all_e = torch.stack(list_e, 1)
        all_is_obj = torch.stack(list_is_obj, 1)
        return all_e, all_is_obj


class TwoHeads(nn.Module):

    def __init__(self, num_classes=174, cnn=None, features_size=512, time=8, mask_size=28, size_RN=256, nb_head=2, logits_type='object', f_orn=True, size_2nd_head=14, **kwargs):
        super(TwoHeads, self).__init__()
        self.num_classes = num_classes
        self.time = time
        self.size_cnn_features = features_size
        self.size_mask = mask_size
        self.size_RN = size_RN
        self.size_RNN = size_RN
        self.nb_head = nb_head
        self.logits_type = logits_type
        self.size_fm_2nd_head = size_2nd_head
        self.cnn = cnn
        self.cnn.out_dim = 5
        if self.logits_type == 'object':
            for i, child in enumerate(self.cnn.children()):
                if i != 7:
                    for param in child.parameters():
                        param.requires_grad = False
        self.avgpool_TxMxM = nn.AvgPool3d((self.time, self.size_fm_2nd_head, self.size_fm_2nd_head))
        self.avgpool_1xMxM = nn.AvgPool3d((1, self.size_fm_2nd_head, self.size_fm_2nd_head))
        self.avgpool_1_7x7 = nn.AvgPool3d((1, 7, 7))
        self.avgpool_T_7x7 = nn.AvgPool3d((self.time, 7, 7))
        self.pixel_pooler = nn.AvgPool3d((1, 2, 2)) if self.size_fm_2nd_head == 14 else nn.AvgPool3d((1, 4, 4))
        self.pool_orn = nn.MaxPool2d((self.time - 1, 1))
        self.size_COCO_object_features = self.size_cnn_features
        self.COCO_Object_Class_from_Features = Classifier(size_input=self.size_cnn_features, size_output=81)
        self.size_mask_embedding = 100
        self.Encoder_Binary_Mask = EncoderMLP(input_size=self.size_mask * self.size_mask, list_hidden_size=[self.size_mask_embedding, self.size_mask_embedding])
        self.size_obj_embedding = 100
        input_size = 81
        self.Encoder_COCO_Obj_Class = EncoderMLP(input_size=input_size, list_hidden_size=[self.size_obj_embedding, self.size_obj_embedding])
        size_object = self.size_COCO_object_features + self.size_mask_embedding + self.size_obj_embedding
        list_size_hidden_layers = [self.size_RN, self.size_RN, self.size_RN]
        self.ORN = ObjectRelationNetwork(size_object=size_object, list_hidden_layers_size=list_size_hidden_layers)
        self.AggregationRelations = AggregationRelations()
        self.f_orn = f_orn
        if self.f_orn == 'rnn':
            input_size = self.size_RN
            self.size_relation_features = int(self.size_RN / 2.0)
            self.rnn_objects = nn.GRU(input_size=input_size, hidden_size=self.size_relation_features, num_layers=1, batch_first=True)
        else:
            self.size_relation_features = self.size_RN
        self.fc_classifier_object = nn.Linear(self.size_relation_features, self.num_classes)
        self.fc_classifier_context = nn.Linear(self.size_cnn_features, self.num_classes)

    def get_objects_features(self, fm, masks):
        fm = fm.transpose(1, 2)
        B, T, D, W, H = fm.size()
        fm = fm.view(B * T, D, W, H)
        fm_up = F.upsample(fm, size=(self.size_mask, self.size_mask), mode='bilinear', align_corners=True)
        fm_up = fm_up.view(B, T, D, self.size_mask, self.size_mask)
        fm_up = fm_up.transpose(1, 2)
        B, D, T, W, _ = fm_up.size()
        K = masks.size(2)
        W_masks = masks.size(-1)
        fm_plus = fm_up.unsqueeze(1)
        masks_plus = masks.transpose(1, 2)
        masks_plus = masks_plus.unsqueeze(2)
        fm_masked = fm_plus * masks_plus
        masks_size = torch.sum(torch.sum(masks_plus, -1), -1)
        list_object_set = []
        for t in range(T):
            object_set_avg = torch.sum(torch.sum(fm_masked[:, :, :, (t)], -1), -1)
            object_set_avg /= masks_size[:, :, :, (t)] + 0.0001
            list_object_set.append(object_set_avg)
        objects_features = torch.stack(list_object_set, 1)
        return objects_features

    def get_pixel_features(self, fm):
        *_, W, H = fm.size()
        fm = fm.transpose(1, 2)
        list_pixel_features = []
        if W != 7:
            fm = self.pixel_pooler(fm)
            *_, W, H = fm.size()
        for i in range(W):
            for j in range(H):
                list_pixel_features.append(fm[:, :, :, (i), (j)])
        pixel_features = torch.stack(list_pixel_features, 2)
        return pixel_features

    def retrieve_relations(self, relational_reasoning_vector_COCO, obj_id):
        B, T, K, C = obj_id.size()
        B, T_1, K2, D = relational_reasoning_vector_COCO.size()
        relations = np.zeros((B, 81, 81))
        for b in range(B):
            for t in range(T_1):
                for k2 in range(K2):
                    try:
                        inter = torch.sum(relational_reasoning_vector_COCO[b, t, k2])
                        k_1 = math.floor(float(k2) / float(K))
                        obj_id_k_1 = self.get_id_object(obj_id[b, t, k_1].data.cpu().numpy())
                        k = k2 - k_1 * K
                        obj_id_k = self.get_id_object(obj_id[b, t + 1, k].data.cpu().numpy())
                        relations[b, obj_id_k_1, obj_id_k] = inter
                    except:
                        pass
        return relations

    def retrieve_relations_temporal(self, relational_reasoning_vector_COCO, obj_id):
        B, T, K, C = obj_id.size()
        B, T_1, K2, D = relational_reasoning_vector_COCO.size()
        relations = np.zeros((B, T_1, 81, 81))
        for b in range(B):
            for t in range(T_1):
                for k2 in range(K2):
                    try:
                        inter = torch.sum(relational_reasoning_vector_COCO[b, t, k2])
                        k_1 = math.floor(float(k2) / float(K))
                        obj_id_k_1 = self.get_id_object(obj_id[b, t, k_1].data.cpu().numpy())
                        k = k2 - k_1 * K
                        obj_id_k = self.get_id_object(obj_id[b, t + 1, k].data.cpu().numpy())
                        relations[b, t, obj_id_k_1, obj_id_k] = inter
                    except:
                        pass
        return relations

    @staticmethod
    def get_id_object(one_hot):
        if one_hot[0] == 1:
            return 0
        else:
            return np.argmax(one_hot)

    def context_head(self, fm_context, B):
        context_vector = self.avgpool_T_7x7(fm_context)
        context_representation = context_vector.view(B, self.size_cnn_features)
        return context_representation

    def object_head(self, fm_objects, masks, obj_id, B):
        objects_features = self.get_objects_features(fm_objects, masks)
        preds_class_detected_objects = self.COCO_Object_Class_from_Features(objects_features)
        embedding_objects_location = self.Encoder_Binary_Mask(masks)
        embedding_obj_id = self.Encoder_COCO_Obj_Class(obj_id)
        full_objects = torch.cat([objects_features, embedding_objects_location, embedding_obj_id], -1)
        D = self.size_cnn_features
        all_e, all_is_obj = self.ORN(full_objects, D)
        all_is_obj = all_is_obj.unsqueeze(-1)
        all_e *= all_is_obj
        orn_aggregated = self.AggregationRelations(all_e)
        if self.f_orn == 'rnn':
            object_representation, _ = self.rnn_objects(orn_aggregated)
            object_representation = torch.mean(object_representation, 1)
        elif self.f_orn == 'avg':
            object_representation = torch.mean(orn_aggregated, 1)
        else:
            raise Exception
        return object_representation, preds_class_detected_objects

    def add(self, logits, logits_head):
        if logits is None:
            return logits_head
        else:
            return logits + logits_head

    def squeeze_masks(self, max_nb_obj, masks, obj_id, bbox):
        nb_max_obj_in_B = int(torch.max(max_nb_obj).cpu().numpy())
        masks_squeezed = masks[:, :, :nb_max_obj_in_B]
        obj_id_squeezed = obj_id[:, :, :nb_max_obj_in_B]
        bbox_squeezed = bbox[:, :, :nb_max_obj_in_B]
        return masks_squeezed, obj_id_squeezed, bbox_squeezed

    def final_classification(self, context_representation, object_representation):
        if context_representation is not None:
            return self.fc_classifier_object(object_representation) + self.fc_classifier_context(context_representation)
        else:
            return self.fc_classifier_object(object_representation)

    def forward(self, x):
        """Forward pass from a tensor of size (B,C,T,W,H)"""
        clip, masks, obj_id, bbox, max_nb_obj = x['clip'], x['mask'], x['obj_id'], x['obj_bbox'], x['max_nb_obj']
        masks, obj_id, bbox = self.squeeze_masks(max_nb_obj, masks, obj_id, bbox)
        B = clip.size(0)
        T = clip.size(2)
        K = masks.size(2)
        fm_context, fm_objects = self.cnn.get_two_heads_feature_maps(x, T=T, out_dim=5, heads_type=self.logits_type)
        context_representation, object_representation, preds_class_detected_objects = None, None, None
        if 'object' in self.logits_type:
            object_representation, preds_class_detected_objects = self.object_head(fm_objects, masks, obj_id, B=B)
        if 'context' in self.logits_type:
            context_representation = self.context_head(fm_context, B=B)
        logits = self.final_classification(context_representation, object_representation)
        return logits, preds_class_detected_objects


class CriterionLinearCombination(Module):

    def __init__(self, list_criterion_names, list_weights):
        super(CriterionLinearCombination, self).__init__()
        assert len(list_criterion_names) == len(list_weights)
        self.list_criterion, self.list_weights = [], []
        for i, criterion_name in enumerate(list_criterion_names):
            if criterion_name == 'bce':
                self.list_criterion.append(nn.BCEWithLogitsLoss())
            elif criterion_name == 'ce':
                self.list_criterion.append(nn.CrossEntropyLoss())
            else:
                raise Exception
            self.list_weights.append(list_weights[i])

    def forward(self, list_input, list_target, cuda=False):
        assert len(list_input) == len(list_target)
        loss = 0.0
        for i in range(len(self.list_criterion)):
            criterion_i, weight_i = self.list_criterion[i], self.list_weights[i]
            target_i, input_i = list_target[i], list_input[i]
            if input_i is not None:
                if isinstance(criterion_i, nn.CrossEntropyLoss):
                    target_i = target_i.type(torch.LongTensor)
                elif isinstance(criterion_i, nn.BCEWithLogitsLoss):
                    target_i = target_i.type(torch.FloatTensor)
                target_i = target_i if cuda else target_i
                input_i = input_i.view(-1, input_i.size(-1))
                loss_i = weight_i * criterion_i(input_i, target_i)
                loss = loss + loss_i
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AggregationRelations,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock2D,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock2_1D,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     True),
    (BasicBlock3D,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     True),
    (Classifier,
     lambda: ([], {'size_input': 4, 'size_output': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_fabienbaradel_object_level_visual_reasoning(_paritybench_base):
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

