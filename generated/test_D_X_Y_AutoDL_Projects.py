import sys
_module = sys.modules[__name__]
del sys
GDAS = _module
check = _module
functions = _module
main = _module
statistics = _module
visualize = _module
xshapes = _module
BOHB = _module
ENAS = _module
GDAS = _module
RANDOM = _module
R_EA = _module
SETN = _module
reinforce = _module
prepare = _module
test = _module
config_utils = _module
attention_args = _module
basic_args = _module
cls_init_args = _module
cls_kd_args = _module
configure_utils = _module
pruning_args = _module
random_baseline = _module
search_args = _module
search_single_args = _module
share_args = _module
DownsampledImageNet = _module
LandmarkDataset = _module
SearchDatasetWrap = _module
datasets = _module
get_dataset_with_transform = _module
landmark_utils = _module
point_meta = _module
test_utils = _module
log_utils = _module
logger = _module
meter = _module
time_utils = _module
CifarDenseNet = _module
CifarResNet = _module
CifarWideResNet = _module
ImageNet_MobileNetV2 = _module
ImageNet_ResNet = _module
SharedUtils = _module
models = _module
cell_infers = _module
cells = _module
nasnet_cifar = _module
tiny_network = _module
cell_operations = _module
cell_searchs = _module
_test_module = _module
genotypes = _module
search_cells = _module
search_model_darts = _module
search_model_darts_nasnet = _module
search_model_enas = _module
search_model_enas_utils = _module
search_model_gdas = _module
search_model_gdas_nasnet = _module
search_model_random = _module
search_model_setn = _module
search_model_setn_nasnet = _module
clone_weights = _module
initialization = _module
InferCifarResNet = _module
InferCifarResNet_depth = _module
InferCifarResNet_width = _module
InferImagenetResNet = _module
InferMobileNetV2 = _module
InferTinyCellNet = _module
shape_infers = _module
shared_utils = _module
SearchCifarResNet = _module
SearchCifarResNet_depth = _module
SearchCifarResNet_width = _module
SearchImagenetResNet = _module
SearchSimResNet_width = _module
SoftSelect = _module
shape_searchs = _module
test = _module
nas_201_api = _module
api = _module
CifarNet = _module
ImageNet = _module
DXYs = _module
base_cells = _module
construct_utils = _module
head_utils = _module
nas_infer_model = _module
operations = _module
procedures = _module
basic_main = _module
funcs_nasbench = _module
optimizers = _module
search_main = _module
search_main_v2 = _module
simple_KD_main = _module
starts = _module
tf_models = _module
tf_optimizers = _module
weight_decay_optimizers = _module
utils = _module
affine_utils = _module
evaluation_utils = _module
flop_benchmark = _module
gpu_manager = _module
nas_utils = _module
weight_watcher = _module

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


import time


import torch


import random


from copy import deepcopy


import numpy as np


import torch.nn as nn


import collections


from torch.distributions import Categorical


from collections import OrderedDict


import math


import torch.nn.functional as F


from torch import nn


import warnings


from typing import List


from typing import Text


from typing import Dict


from torch.distributions.categorical import Categorical


from typing import Any


import copy


from torch.optim import Optimizer


class Policy(nn.Module):

    def __init__(self, max_nodes, search_space):
        super(Policy, self).__init__()
        self.max_nodes = max_nodes
        self.search_space = deepcopy(search_space)
        self.edge2index = {}
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                self.edge2index[node_str] = len(self.edge2index)
        self.arch_parameters = nn.Parameter(0.001 * torch.randn(len(self.edge2index), len(search_space)))

    def generate_arch(self, actions):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                op_name = self.search_space[actions[self.edge2index[node_str]]]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    def genotype(self):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    weights = self.arch_parameters[self.edge2index[node_str]]
                    op_name = self.search_space[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    def forward(self):
        alphas = nn.functional.softmax(self.arch_parameters, dim=-1)
        return alphas


class Bottleneck(nn.Module):

    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):

    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):

    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


def initialize_resnet(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


class DenseNet(nn.Module):

    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()
        if bottleneck:
            nDenseBlocks = int((depth - 4) / 6)
        else:
            nDenseBlocks = int((depth - 4) / 3)
        self.message = 'CifarDenseNet : block : {:}, depth : {:}, reduction : {:}, growth-rate = {:}, class = {:}'.format('bottleneck' if bottleneck else 'basic', depth, reduction, growthRate, nClasses)
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        self.act = nn.Sequential(nn.BatchNorm2d(nChannels), nn.ReLU(inplace=True), nn.AvgPool2d(8))
        self.fc = nn.Linear(nChannels, nClasses)
        self.apply(initialize_resnet)

    def get_message(self):
        return self.message

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        features = self.act(out)
        features = features.view(features.size(0), -1)
        out = self.fc(features)
        return features, out


class Downsample(nn.Module):

    def __init__(self, nIn, nOut, stride):
        super(Downsample, self).__init__()
        assert stride == 2 and nOut == 2 * nIn, 'stride:{} IO:{},{}'.format(stride, nIn, nOut)
        self.in_dim = nIn
        self.out_dim = nOut
        self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.avg(x)
        out = self.conv(x)
        return out


class ConvBNReLU(nn.Module):

    def __init__(self, nIn, nOut, kernel, stride, padding, bias, relu):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(nOut)
        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None
        self.out_dim = nOut
        self.num_conv = 1

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        if self.relu:
            return self.relu(bn)
        else:
            return bn


def additive_func(A, B):
    assert A.dim() == B.dim() and A.size(0) == B.size(0), '{:} vs {:}'.format(A.size(), B.size())
    C = min(A.size(1), B.size(1))
    if A.size(1) == B.size(1):
        return A + B
    elif A.size(1) < B.size(1):
        out = B.clone()
        out[:, :C] += A
        return out
    else:
        out = A.clone()
        out[:, :C] += B
        return out


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ConvBNReLU(inplanes, planes, 3, stride, 1, False, True)
        self.conv_b = ConvBNReLU(planes, planes, 3, 1, 1, False, False)
        if stride == 2:
            self.downsample = Downsample(inplanes, planes, stride)
        elif inplanes != planes:
            self.downsample = ConvBNReLU(inplanes, planes, 1, 1, 0, False, False)
        else:
            self.downsample = None
        self.out_dim = planes
        self.num_conv = 2

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = additive_func(residual, basicblock)
        return F.relu(out, inplace=True)


class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride):
        super(ResNetBottleneck, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_1x1 = ConvBNReLU(inplanes, planes, 1, 1, 0, False, True)
        self.conv_3x3 = ConvBNReLU(planes, planes, 3, stride, 1, False, True)
        self.conv_1x4 = ConvBNReLU(planes, planes * self.expansion, 1, 1, 0, False, False)
        if stride == 2:
            self.downsample = Downsample(inplanes, planes * self.expansion, stride)
        elif inplanes != planes * self.expansion:
            self.downsample = ConvBNReLU(inplanes, planes * self.expansion, 1, 1, 0, False, False)
        else:
            self.downsample = None
        self.out_dim = planes * self.expansion
        self.num_conv = 3

    def forward(self, inputs):
        bottleneck = self.conv_1x1(inputs)
        bottleneck = self.conv_3x3(bottleneck)
        bottleneck = self.conv_1x4(bottleneck)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = additive_func(residual, bottleneck)
        return F.relu(out, inplace=True)


class CifarResNet(nn.Module):

    def __init__(self, block_name, depth, num_classes, zero_init_residual):
        super(CifarResNet, self).__init__()
        if block_name == 'ResNetBasicblock':
            block = ResNetBasicblock
            assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
            layer_blocks = (depth - 2) // 6
        elif block_name == 'ResNetBottleneck':
            block = ResNetBottleneck
            assert (depth - 2) % 9 == 0, 'depth should be one of 164'
            layer_blocks = (depth - 2) // 9
        else:
            raise ValueError('invalid block : {:}'.format(block_name))
        self.message = 'CifarResNet : Block : {:}, Depth : {:}, Layers for each block : {:}'.format(block_name, depth, layer_blocks)
        self.num_classes = num_classes
        self.channels = [16]
        self.layers = nn.ModuleList([ConvBNReLU(3, 16, 3, 1, 1, False, True)])
        for stage in range(3):
            for iL in range(layer_blocks):
                iC = self.channels[-1]
                planes = 16 * 2 ** stage
                stride = 2 if stage > 0 and iL == 0 else 1
                module = block(iC, planes, stride)
                self.channels.append(module.out_dim)
                self.layers.append(module)
                self.message += '\nstage={:}, ilayer={:02d}/{:02d}, block={:03d}, iC={:3d}, oC={:3d}, stride={:}'.format(stage, iL, layer_blocks, len(self.layers) - 1, iC, module.out_dim, stride)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(module.out_dim, num_classes)
        assert sum(x.num_conv for x in self.layers) + 1 == depth, 'invalid depth check {:} vs {:}'.format(sum(x.num_conv for x in self.layers) + 1, depth)
        self.apply(initialize_resnet)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNetBasicblock):
                    nn.init.constant_(m.conv_b.bn.weight, 0)
                elif isinstance(m, ResNetBottleneck):
                    nn.init.constant_(m.conv_1x4.bn.weight, 0)

    def get_message(self):
        return self.message

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return features, logits


class WideBasicblock(nn.Module):

    def __init__(self, inplanes, planes, stride, dropout=False):
        super(WideBasicblock, self).__init__()
        self.bn_a = nn.BatchNorm2d(inplanes)
        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)
        if dropout:
            self.dropout = nn.Dropout2d(p=0.5, inplace=True)
        else:
            self.dropout = None
        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if inplanes != planes:
            self.downsample = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        basicblock = self.bn_a(x)
        basicblock = F.relu(basicblock)
        basicblock = self.conv_a(basicblock)
        basicblock = self.bn_b(basicblock)
        basicblock = F.relu(basicblock)
        if self.dropout is not None:
            basicblock = self.dropout(basicblock)
        basicblock = self.conv_b(basicblock)
        if self.downsample is not None:
            x = self.downsample(x)
        return x + basicblock


class CifarWideResNet(nn.Module):
    """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """

    def __init__(self, depth, widen_factor, num_classes, dropout):
        super(CifarWideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 4) // 6
        None
        self.num_classes = num_classes
        self.dropout = dropout
        self.conv_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.message = 'Wide ResNet : depth={:}, widen_factor={:}, class={:}'.format(depth, widen_factor, num_classes)
        self.inplanes = 16
        self.stage_1 = self._make_layer(WideBasicblock, 16 * widen_factor, layer_blocks, 1)
        self.stage_2 = self._make_layer(WideBasicblock, 32 * widen_factor, layer_blocks, 2)
        self.stage_3 = self._make_layer(WideBasicblock, 64 * widen_factor, layer_blocks, 2)
        self.lastact = nn.Sequential(nn.BatchNorm2d(64 * widen_factor), nn.ReLU(inplace=True))
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64 * widen_factor, num_classes)
        self.apply(initialize_resnet)

    def get_message(self):
        return self.message

    def _make_layer(self, block, planes, blocks, stride):
        layers = []
        layers.append(block(self.inplanes, planes, stride, self.dropout))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, self.dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_3x3(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.lastact(x)
        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        outs = self.classifier(features)
        return features, outs


class ConvBNReLU(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):

    def __init__(self, num_classes, width_mult, input_channel, last_channel, block_name, dropout):
        super(MobileNetV2, self).__init__()
        if block_name == 'InvertedResidual':
            block = InvertedResidual
        else:
            raise ValueError('invalid block name : {:}'.format(block_name))
        inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.last_channel, num_classes))
        self.message = 'MobileNetV2 : width_mult={:}, in-C={:}, last-C={:}, block={:}, dropout={:}'.format(width_mult, input_channel, last_channel, block_name, dropout)
        self.apply(initialize_resnet)

    def get_message(self):
        return self.message

    def forward(self, inputs):
        features = self.features(inputs)
        vectors = features.mean([2, 3])
        predicts = self.classifier(vectors)
        return features, predicts


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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

    def __init__(self, block_name, layers, deep_stem, num_classes, zero_init_residual, groups, width_per_group):
        super(ResNet, self).__init__()
        if block_name == 'BasicBlock':
            block = BasicBlock
        elif block_name == 'Bottleneck':
            block = Bottleneck
        else:
            raise ValueError('invalid block-name : {:}'.format(block_name))
        if not deep_stem:
            self.conv = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.inplanes = 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, groups=groups, base_width=width_per_group)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, groups=groups, base_width=width_per_group)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, groups=groups, base_width=width_per_group)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, groups=groups, base_width=width_per_group)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.message = 'block = {:}, layers = {:}, deep_stem = {:}, num_classes = {:}'.format(block, layers, deep_stem, num_classes)
        self.apply(initialize_resnet)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride, groups, base_width):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 2:
                downsample = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0), conv1x1(self.inplanes, planes * block.expansion, 1), nn.BatchNorm2d(planes * block.expansion))
            elif stride == 1:
                downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
            else:
                raise ValueError('invalid stride [{:}] for downsample'.format(stride))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, base_width))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, groups, base_width))
        return nn.Sequential(*layers)

    def get_message(self):
        return self.message

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = self.fc(features)
        return features, logits


OPS = {'none': lambda C_in, C_out, stride, affine: Zero(stride), 'avg_pool_3x3': lambda C_in, C_out, stride, affine: POOLING(C_in, C_out, stride, 'avg'), 'max_pool_3x3': lambda C_in, C_out, stride, affine: POOLING(C_in, C_out, stride, 'max'), 'nor_conv_7x7': lambda C_in, C_out, stride, affine: ReLUConvBN(C_in, C_out, (7, 7), (stride, stride), (3, 3), affine), 'nor_conv_3x3': lambda C_in, C_out, stride, affine: ReLUConvBN(C_in, C_out, (3, 3), (stride, stride), (1, 1), affine), 'nor_conv_1x1': lambda C_in, C_out, stride, affine: ReLUConvBN(C_in, C_out, (1, 1), (stride, stride), (0, 0), affine), 'skip_connect': lambda C_in, C_out, stride, affine: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride, affine), 'sep_conv_3x3': lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 3, stride, 1, affine=affine), 'sep_conv_5x5': lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 5, stride, 2, affine=affine), 'sep_conv_7x7': lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 7, stride, 3, affine=affine), 'dil_conv_3x3': lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 3, stride, 2, 2, affine=affine), 'dil_conv_5x5': lambda C_in, C_out, stride, affine: DilConv(C_in, C_out, 5, stride, 4, 2, affine=affine), 'conv_7x1_1x7': lambda C_in, C_out, stride, affine: Conv717(C_in, C_out, stride, affine), 'conv_3x1_1x3': lambda C_in, C_out, stride, affine: Conv313(C_in, C_out, stride, affine)}


class InferCell(nn.Module):

    def __init__(self, genotype, C_in, C_out, stride):
        super(InferCell, self).__init__()
        self.layers = nn.ModuleList()
        self.node_IN = []
        self.node_IX = []
        self.genotype = deepcopy(genotype)
        for i in range(1, len(genotype)):
            node_info = genotype[i - 1]
            cur_index = []
            cur_innod = []
            for op_name, op_in in node_info:
                if op_in == 0:
                    layer = OPS[op_name](C_in, C_out, stride, True, True)
                else:
                    layer = OPS[op_name](C_out, C_out, 1, True, True)
                cur_index.append(len(self.layers))
                cur_innod.append(op_in)
                self.layers.append(layer)
            self.node_IX.append(cur_index)
            self.node_IN.append(cur_innod)
        self.nodes = len(genotype)
        self.in_dim = C_in
        self.out_dim = C_out

    def extra_repr(self):
        string = 'info :: nodes={nodes}, inC={in_dim}, outC={out_dim}'.format(**self.__dict__)
        laystr = []
        for i, (node_layers, node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
            y = ['I{:}-L{:}'.format(_ii, _il) for _il, _ii in zip(node_layers, node_innods)]
            x = '{:}<-({:})'.format(i + 1, ','.join(y))
            laystr.append(x)
        return string + ', [{:}]'.format(' | '.join(laystr)) + ', {:}'.format(self.genotype.tostr())

    def forward(self, inputs):
        nodes = [inputs]
        for i, (node_layers, node_innods) in enumerate(zip(self.node_IX, self.node_IN)):
            node_feature = sum(self.layers[_il](nodes[_ii]) for _il, _ii in zip(node_layers, node_innods))
            nodes.append(node_feature)
        return nodes[-1]


class NASNetInferCell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, affine, track_running_stats):
        super(NASNetInferCell, self).__init__()
        self.reduction = reduction
        if reduction_prev:
            self.preprocess0 = OPS['skip_connect'](C_prev_prev, C, 2, affine, track_running_stats)
        else:
            self.preprocess0 = OPS['nor_conv_1x1'](C_prev_prev, C, 1, affine, track_running_stats)
        self.preprocess1 = OPS['nor_conv_1x1'](C_prev, C, 1, affine, track_running_stats)
        if not reduction:
            nodes, concats = genotype['normal'], genotype['normal_concat']
        else:
            nodes, concats = genotype['reduce'], genotype['reduce_concat']
        self._multiplier = len(concats)
        self._concats = concats
        self._steps = len(nodes)
        self._nodes = nodes
        self.edges = nn.ModuleDict()
        for i, node in enumerate(nodes):
            for in_node in node:
                name, j = in_node[0], in_node[1]
                stride = 2 if reduction and j < 2 else 1
                node_str = '{:}<-{:}'.format(i + 2, j)
                self.edges[node_str] = OPS[name](C, C, stride, affine, track_running_stats)

    def forward(self, s0, s1, unused_drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i, node in enumerate(self._nodes):
            clist = []
            for in_node in node:
                name, j = in_node[0], in_node[1]
                node_str = '{:}<-{:}'.format(i + 2, j)
                op = self.edges[node_str]
                clist.append(op(states[j]))
            states.append(sum(clist))
        return torch.cat([states[x] for x in self._concats], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), nn.Conv2d(C, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 768, 2, bias=False), nn.BatchNorm2d(768), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NASNetonCIFAR(nn.Module):

    def __init__(self, C, N, stem_multiplier, num_classes, genotype, auxiliary, affine=True, track_running_stats=True):
        super(NASNetonCIFAR, self).__init__()
        self._C = C
        self._layerN = N
        self.stem = nn.Sequential(nn.Conv2d(3, C * stem_multiplier, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C * stem_multiplier))
        layer_channels = [C] * N + [C * 2] + [C * 2] * (N - 1) + [C * 4] + [C * 4] * (N - 1)
        layer_reductions = [False] * N + [True] + [False] * (N - 1) + [True] + [False] * (N - 1)
        C_prev_prev, C_prev, C_curr, reduction_prev = C * stem_multiplier, C * stem_multiplier, C, False
        self.auxiliary_index = None
        self.auxiliary_head = None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            cell = InferCell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, affine, track_running_stats)
            self.cells.append(cell)
            C_prev_prev, C_prev, reduction_prev = C_prev, cell._multiplier * C_curr, reduction
            if reduction and C_curr == C * 4 and auxiliary:
                self.auxiliary_head = AuxiliaryHeadCIFAR(C_prev, num_classes)
                self.auxiliary_index = index
        self._Layer = len(self.cells)
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.drop_path_prob = -1

    def update_drop_path(self, drop_path_prob):
        self.drop_path_prob = drop_path_prob

    def auxiliary_param(self):
        if self.auxiliary_head is None:
            return []
        else:
            return list(self.auxiliary_head.parameters())

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return '{name}(C={_C}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs):
        stem_feature, logits_aux = self.stem(inputs), None
        cell_results = [stem_feature, stem_feature]
        for i, cell in enumerate(self.cells):
            cell_feature = cell(cell_results[-2], cell_results[-1], self.drop_path_prob)
            cell_results.append(cell_feature)
            if self.auxiliary_index is not None and i == self.auxiliary_index and self.training:
                logits_aux = self.auxiliary_head(cell_results[-1])
        out = self.lastact(cell_results[-1])
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        if logits_aux is None:
            return out, logits
        else:
            return out, [logits, logits_aux]


class TinyNetwork(nn.Module):

    def __init__(self, C, N, genotype, num_classes):
        super(TinyNetwork, self).__init__()
        self._C = C
        self._layerN = N
        self.stem = nn.Sequential(nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C))
        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
        C_prev = C
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
            else:
                cell = InferCell(genotype, C_prev, C_curr, 1)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self._Layer = len(self.cells)
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return '{name}(C={_C}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False), nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats))

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats))

    def forward(self, x):
        return self.op(x)


class DualSepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
        super(DualSepConv, self).__init__()
        self.op_a = SepConv(C_in, C_in, kernel_size, stride, padding, dilation, affine, track_running_stats)
        self.op_b = SepConv(C_in, C_out, kernel_size, 1, padding, dilation, affine, track_running_stats)

    def forward(self, x):
        x = self.op_a(x)
        x = self.op_b(x)
        return x


class ResNetBasicblock(nn.Module):

    def __init__(self, inplanes, planes, stride, affine=True):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, affine)
        self.conv_b = ReLUConvBN(planes, planes, 3, 1, 1, 1, affine)
        if stride == 2:
            self.downsample = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0), nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1, affine)
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def extra_repr(self):
        string = '{name}(inC={in_dim}, outC={out_dim}, stride={stride})'.format(name=self.__class__.__name__, **self.__dict__)
        return string

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock


class POOLING(nn.Module):

    def __init__(self, C_in, C_out, stride, mode, affine=True, track_running_stats=True):
        super(POOLING, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 1, affine, track_running_stats)
        if mode == 'avg':
            self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == 'max':
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError('Invalid mode={:} in POOLING'.format(mode))

    def forward(self, inputs):
        if self.preprocess:
            x = self.preprocess(inputs)
        else:
            x = inputs
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            else:
                return x[:, :, ::self.stride, ::self.stride].mul(0.0)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, stride, affine, track_running_stats):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False))
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 1:
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False)
        else:
            raise ValueError('Invalid stride : {:}'.format(stride))
        self.bn = nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        if self.stride == 2:
            x = self.relu(x)
            y = self.pad(x)
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        else:
            out = self.conv(x)
        out = self.bn(out)
        return out

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


class PartAwareOp(nn.Module):

    def __init__(self, C_in, C_out, stride, part=4):
        super().__init__()
        self.part = 4
        self.hidden = C_in // 3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.local_conv_list = nn.ModuleList()
        for i in range(self.part):
            self.local_conv_list.append(nn.Sequential(nn.ReLU(), nn.Conv2d(C_in, self.hidden, 1), nn.BatchNorm2d(self.hidden, affine=True)))
        self.W_K = nn.Linear(self.hidden, self.hidden)
        self.W_Q = nn.Linear(self.hidden, self.hidden)
        if stride == 2:
            self.last = FactorizedReduce(C_in + self.hidden, C_out, 2)
        elif stride == 1:
            self.last = FactorizedReduce(C_in + self.hidden, C_out, 1)
        else:
            raise ValueError('Invalid Stride : {:}'.format(stride))

    def forward(self, x):
        batch, C, H, W = x.size()
        assert H >= self.part, 'input size too small : {:} vs {:}'.format(x.shape, self.part)
        IHs = [0]
        for i in range(self.part):
            IHs.append(min(H, int((i + 1) * (float(H) / self.part))))
        local_feat_list = []
        for i in range(self.part):
            feature = x[:, :, IHs[i]:IHs[i + 1], :]
            xfeax = self.avg_pool(feature)
            xfea = self.local_conv_list[i](xfeax)
            local_feat_list.append(xfea)
        part_feature = torch.cat(local_feat_list, dim=2).view(batch, -1, self.part)
        part_feature = part_feature.transpose(1, 2).contiguous()
        part_K = self.W_K(part_feature)
        part_Q = self.W_Q(part_feature).transpose(1, 2).contiguous()
        weight_att = torch.bmm(part_K, part_Q)
        attention = torch.softmax(weight_att, dim=2)
        aggreateF = torch.bmm(attention, part_feature).transpose(1, 2).contiguous()
        features = []
        for i in range(self.part):
            feature = aggreateF[:, :, i:i + 1].expand(batch, self.hidden, IHs[i + 1] - IHs[i])
            feature = feature.view(batch, self.hidden, IHs[i + 1] - IHs[i], 1)
            features.append(feature)
        features = torch.cat(features, dim=2).expand(batch, self.hidden, H, W)
        final_fea = torch.cat((x, features), dim=1)
        outputs = self.last(final_fea)
        return outputs


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = x.new_zeros(x.size(0), 1, 1, 1)
        mask = mask.bernoulli_(keep_prob)
        x = torch.div(x, keep_prob)
        x.mul_(mask)
    return x


class GDAS_Reduction_Cell(nn.Module):

    def __init__(self, C_prev_prev, C_prev, C, reduction_prev, multiplier, affine, track_running_stats):
        super(GDAS_Reduction_Cell, self).__init__()
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, 2, affine, track_running_stats)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, 1, affine, track_running_stats)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, 1, affine, track_running_stats)
        self.multiplier = multiplier
        self.reduction = True
        self.ops1 = nn.ModuleList([nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C, C, (1, 3), stride=(1, 2), padding=(0, 1), groups=8, bias=False), nn.Conv2d(C, C, (3, 1), stride=(2, 1), padding=(1, 0), groups=8, bias=False), nn.BatchNorm2d(C, affine=True), nn.ReLU(inplace=False), nn.Conv2d(C, C, 1, stride=1, padding=0, bias=False), nn.BatchNorm2d(C, affine=True)), nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C, C, (1, 3), stride=(1, 2), padding=(0, 1), groups=8, bias=False), nn.Conv2d(C, C, (3, 1), stride=(2, 1), padding=(1, 0), groups=8, bias=False), nn.BatchNorm2d(C, affine=True), nn.ReLU(inplace=False), nn.Conv2d(C, C, 1, stride=1, padding=0, bias=False), nn.BatchNorm2d(C, affine=True))])
        self.ops2 = nn.ModuleList([nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.BatchNorm2d(C, affine=True)), nn.Sequential(nn.MaxPool2d(3, stride=2, padding=1), nn.BatchNorm2d(C, affine=True))])

    def forward(self, s0, s1, drop_prob=-1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        X0 = self.ops1[0](s0)
        X1 = self.ops1[1](s1)
        if self.training and drop_prob > 0.0:
            X0, X1 = drop_path(X0, drop_prob), drop_path(X1, drop_prob)
        X2 = self.ops2[0](s0)
        X3 = self.ops2[1](s1)
        if self.training and drop_prob > 0.0:
            X2, X3 = drop_path(X2, drop_prob), drop_path(X3, drop_prob)
        return torch.cat([X0, X1, X2, X3], dim=1)


class NAS201SearchCell(nn.Module):

    def __init__(self, C_in, C_out, stride, max_nodes, op_names, affine=False, track_running_stats=True):
        super(NAS201SearchCell, self).__init__()
        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                if j == 0:
                    xlists = [OPS[op_name](C_in, C_out, stride, affine, track_running_stats) for op_name in op_names]
                else:
                    xlists = [OPS[op_name](C_in, C_out, 1, affine, track_running_stats) for op_name in op_names]
                self.edges[node_str] = nn.ModuleList(xlists)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def extra_repr(self):
        string = 'info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}'.format(**self.__dict__)
        return string

    def forward(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(sum(layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights)))
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    def forward_gdas(self, inputs, hardwts, index):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                weights = hardwts[self.edge2index[node_str]]
                argmaxs = index[self.edge2index[node_str]].item()
                weigsum = sum(weights[_ie] * edge(nodes[j]) if _ie == argmaxs else weights[_ie] for _ie, edge in enumerate(self.edges[node_str]))
                inter_nodes.append(weigsum)
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    def forward_joint(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                weights = weightss[self.edge2index[node_str]]
                aggregation = sum(layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights))
                inter_nodes.append(aggregation)
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    def forward_urs(self, inputs):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            while True:
                sops, has_non_zero = [], False
                for j in range(i):
                    node_str = '{:}<-{:}'.format(i, j)
                    candidates = self.edges[node_str]
                    select_op = random.choice(candidates)
                    sops.append(select_op)
                    if not hasattr(select_op, 'is_zero') or select_op.is_zero is False:
                        has_non_zero = True
                if has_non_zero:
                    break
            inter_nodes = []
            for j, select_op in enumerate(sops):
                inter_nodes.append(select_op(nodes[j]))
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    def forward_select(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(self.edges[node_str][weights.argmax().item()](nodes[j]))
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    def forward_dynamic(self, inputs, structure):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            cur_op_node = structure.nodes[i - 1]
            inter_nodes = []
            for op_name, j in cur_op_node:
                node_str = '{:}<-{:}'.format(i, j)
                op_index = self.op_names.index(op_name)
                inter_nodes.append(self.edges[node_str][op_index](nodes[j]))
            nodes.append(sum(inter_nodes))
        return nodes[-1]


class MixedOp(nn.Module):

    def __init__(self, space, C, stride, affine, track_running_stats):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in space:
            op = OPS[primitive](C, C, stride, affine, track_running_stats)
            self._ops.append(op)

    def forward_gdas(self, x, weights, index):
        return self._ops[index](x) * weights[index]

    def forward_darts(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class NASNetSearchCell(nn.Module):

    def __init__(self, space, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, affine, track_running_stats):
        super(NASNetSearchCell, self).__init__()
        self.reduction = reduction
        self.op_names = deepcopy(space)
        if reduction_prev:
            self.preprocess0 = OPS['skip_connect'](C_prev_prev, C, 2, affine, track_running_stats)
        else:
            self.preprocess0 = OPS['nor_conv_1x1'](C_prev_prev, C, 1, affine, track_running_stats)
        self.preprocess1 = OPS['nor_conv_1x1'](C_prev, C, 1, affine, track_running_stats)
        self._steps = steps
        self._multiplier = multiplier
        self._ops = nn.ModuleList()
        self.edges = nn.ModuleDict()
        for i in range(self._steps):
            for j in range(2 + i):
                node_str = '{:}<-{:}'.format(i, j)
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(space, C, stride, affine, track_running_stats)
                self.edges[node_str] = op
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def forward_gdas(self, s0, s1, weightss, indexs):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self._steps):
            clist = []
            for j, h in enumerate(states):
                node_str = '{:}<-{:}'.format(i, j)
                op = self.edges[node_str]
                weights = weightss[self.edge2index[node_str]]
                index = indexs[self.edge2index[node_str]].item()
                clist.append(op.forward_gdas(h, weights, index))
            states.append(sum(clist))
        return torch.cat(states[-self._multiplier:], dim=1)

    def forward_darts(self, s0, s1, weightss):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self._steps):
            clist = []
            for j, h in enumerate(states):
                node_str = '{:}<-{:}'.format(i, j)
                op = self.edges[node_str]
                weights = weightss[self.edge2index[node_str]]
                clist.append(op.forward_darts(h, weights))
            states.append(sum(clist))
        return torch.cat(states[-self._multiplier:], dim=1)


def get_combination(space, num):
    combs = []
    for i in range(num):
        if i == 0:
            for func in space:
                combs.append([(func, i)])
        else:
            new_combs = []
            for string in combs:
                for func in space:
                    xstring = string + [(func, i)]
                    new_combs.append(xstring)
            combs = new_combs
    return combs


class Structure:

    def __init__(self, genotype):
        assert isinstance(genotype, list) or isinstance(genotype, tuple), 'invalid class of genotype : {:}'.format(type(genotype))
        self.node_num = len(genotype) + 1
        self.nodes = []
        self.node_N = []
        for idx, node_info in enumerate(genotype):
            assert isinstance(node_info, list) or isinstance(node_info, tuple), 'invalid class of node_info : {:}'.format(type(node_info))
            assert len(node_info) >= 1, 'invalid length : {:}'.format(len(node_info))
            for node_in in node_info:
                assert isinstance(node_in, list) or isinstance(node_in, tuple), 'invalid class of in-node : {:}'.format(type(node_in))
                assert len(node_in) == 2 and node_in[1] <= idx, 'invalid in-node : {:}'.format(node_in)
            self.node_N.append(len(node_info))
            self.nodes.append(tuple(deepcopy(node_info)))

    def tolist(self, remove_str):
        genotypes = []
        for node_info in self.nodes:
            node_info = list(node_info)
            node_info = sorted(node_info, key=lambda x: (x[1], x[0]))
            node_info = tuple(filter(lambda x: x[0] != remove_str, node_info))
            if len(node_info) == 0:
                return None, False
            genotypes.append(node_info)
        return genotypes, True

    def node(self, index):
        assert index > 0 and index <= len(self), 'invalid index={:} < {:}'.format(index, len(self))
        return self.nodes[index]

    def tostr(self):
        strings = []
        for node_info in self.nodes:
            string = '|'.join([(x[0] + '~{:}'.format(x[1])) for x in node_info])
            string = '|{:}|'.format(string)
            strings.append(string)
        return '+'.join(strings)

    def check_valid(self):
        nodes = {(0): True}
        for i, node_info in enumerate(self.nodes):
            sums = []
            for op, xin in node_info:
                if op == 'none' or nodes[xin] is False:
                    x = False
                else:
                    x = True
                sums.append(x)
            nodes[i + 1] = sum(sums) > 0
        return nodes[len(self.nodes)]

    def to_unique_str(self, consider_zero=False):
        nodes = {(0): '0'}
        for i_node, node_info in enumerate(self.nodes):
            cur_node = []
            for op, xin in node_info:
                if consider_zero is None:
                    x = '(' + nodes[xin] + ')' + '@{:}'.format(op)
                elif consider_zero:
                    if op == 'none' or nodes[xin] == '#':
                        x = '#'
                    elif op == 'skip_connect':
                        x = nodes[xin]
                    else:
                        x = '(' + nodes[xin] + ')' + '@{:}'.format(op)
                elif op == 'skip_connect':
                    x = nodes[xin]
                else:
                    x = '(' + nodes[xin] + ')' + '@{:}'.format(op)
                cur_node.append(x)
            nodes[i_node + 1] = '+'.join(sorted(cur_node))
        return nodes[len(self.nodes)]

    def check_valid_op(self, op_names):
        for node_info in self.nodes:
            for inode_edge in node_info:
                if inode_edge[0] not in op_names:
                    return False
        return True

    def __repr__(self):
        return '{name}({node_num} nodes with {node_info})'.format(name=self.__class__.__name__, node_info=self.tostr(), **self.__dict__)

    def __len__(self):
        return len(self.nodes) + 1

    def __getitem__(self, index):
        return self.nodes[index]

    @staticmethod
    def str2structure(xstr):
        assert isinstance(xstr, str), 'must take string (not {:}) as input'.format(type(xstr))
        nodestrs = xstr.split('+')
        genotypes = []
        for i, node_str in enumerate(nodestrs):
            inputs = list(filter(lambda x: x != '', node_str.split('|')))
            for xinput in inputs:
                assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
            inputs = (xi.split('~') for xi in inputs)
            input_infos = tuple((op, int(IDX)) for op, IDX in inputs)
            genotypes.append(input_infos)
        return Structure(genotypes)

    @staticmethod
    def str2fullstructure(xstr, default_name='none'):
        assert isinstance(xstr, str), 'must take string (not {:}) as input'.format(type(xstr))
        nodestrs = xstr.split('+')
        genotypes = []
        for i, node_str in enumerate(nodestrs):
            inputs = list(filter(lambda x: x != '', node_str.split('|')))
            for xinput in inputs:
                assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
            inputs = (xi.split('~') for xi in inputs)
            input_infos = list((op, int(IDX)) for op, IDX in inputs)
            all_in_nodes = list(x[1] for x in input_infos)
            for j in range(i):
                if j not in all_in_nodes:
                    input_infos.append((default_name, j))
            node_info = sorted(input_infos, key=lambda x: (x[1], x[0]))
            genotypes.append(tuple(node_info))
        return Structure(genotypes)

    @staticmethod
    def gen_all(search_space, num, return_ori):
        assert isinstance(search_space, list) or isinstance(search_space, tuple), 'invalid class of search-space : {:}'.format(type(search_space))
        assert num >= 2, 'There should be at least two nodes in a neural cell instead of {:}'.format(num)
        all_archs = get_combination(search_space, 1)
        for i, arch in enumerate(all_archs):
            all_archs[i] = [tuple(arch)]
        for inode in range(2, num):
            cur_nodes = get_combination(search_space, inode)
            new_all_archs = []
            for previous_arch in all_archs:
                for cur_node in cur_nodes:
                    new_all_archs.append(previous_arch + [tuple(cur_node)])
            all_archs = new_all_archs
        if return_ori:
            return all_archs
        else:
            return [Structure(x) for x in all_archs]


class TinyNetworkDarts(nn.Module):

    def __init__(self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats):
        super(TinyNetworkDarts, self).__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C))
        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch_parameters = nn.Parameter(0.001 * torch.randn(num_edge, len(search_space)))

    def get_weights(self):
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(self.global_pooling.parameters())
        xlist += list(self.classifier.parameters())
        return xlist

    def get_alphas(self):
        return [self.arch_parameters]

    def show_alphas(self):
        with torch.no_grad():
            return 'arch-parameters :\n{:}'.format(nn.functional.softmax(self.arch_parameters, dim=-1).cpu())

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return '{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__)

    def genotype(self):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    weights = self.arch_parameters[self.edge2index[node_str]]
                    op_name = self.op_names[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def forward(self, inputs):
        alphas = nn.functional.softmax(self.arch_parameters, dim=-1)
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                feature = cell(feature, alphas)
            else:
                feature = cell(feature)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits


class NASNetworkDARTS(nn.Module):

    def __init__(self, C: int, N: int, steps: int, multiplier: int, stem_multiplier: int, num_classes: int, search_space: List[Text], affine: bool, track_running_stats: bool):
        super(NASNetworkDARTS, self).__init__()
        self._C = C
        self._layerN = N
        self._steps = steps
        self._multiplier = multiplier
        self.stem = nn.Sequential(nn.Conv2d(3, C * stem_multiplier, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C * stem_multiplier))
        layer_channels = [C] * N + [C * 2] + [C * 2] * (N - 1) + [C * 4] + [C * 4] * (N - 1)
        layer_reductions = [False] * N + [True] + [False] * (N - 1) + [True] + [False] * (N - 1)
        num_edge, edge2index = None, None
        C_prev_prev, C_prev, C_curr, reduction_prev = C * stem_multiplier, C * stem_multiplier, C, False
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            cell = SearchCell(search_space, steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, affine, track_running_stats)
            if num_edge is None:
                num_edge, edge2index = cell.num_edges, cell.edge2index
            else:
                assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev_prev, C_prev, reduction_prev = C_prev, multiplier * C_curr, reduction
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch_normal_parameters = nn.Parameter(0.001 * torch.randn(num_edge, len(search_space)))
        self.arch_reduce_parameters = nn.Parameter(0.001 * torch.randn(num_edge, len(search_space)))

    def get_weights(self) ->List[torch.nn.Parameter]:
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(self.global_pooling.parameters())
        xlist += list(self.classifier.parameters())
        return xlist

    def get_alphas(self) ->List[torch.nn.Parameter]:
        return [self.arch_normal_parameters, self.arch_reduce_parameters]

    def show_alphas(self) ->Text:
        with torch.no_grad():
            A = 'arch-normal-parameters :\n{:}'.format(nn.functional.softmax(self.arch_normal_parameters, dim=-1).cpu())
            B = 'arch-reduce-parameters :\n{:}'.format(nn.functional.softmax(self.arch_reduce_parameters, dim=-1).cpu())
        return '{:}\n{:}'.format(A, B)

    def get_message(self) ->Text:
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self) ->Text:
        return '{name}(C={_C}, N={_layerN}, steps={_steps}, multiplier={_multiplier}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__)

    def genotype(self) ->Dict[Text, List]:

        def _parse(weights):
            gene = []
            for i in range(self._steps):
                edges = []
                for j in range(2 + i):
                    node_str = '{:}<-{:}'.format(i, j)
                    ws = weights[self.edge2index[node_str]]
                    for k, op_name in enumerate(self.op_names):
                        if op_name == 'none':
                            continue
                        edges.append((op_name, j, ws[k]))
                edges = sorted(edges, key=lambda x: -x[-1])
                selected_edges = edges[:2]
                gene.append(tuple(selected_edges))
            return gene
        with torch.no_grad():
            gene_normal = _parse(torch.softmax(self.arch_normal_parameters, dim=-1).cpu().numpy())
            gene_reduce = _parse(torch.softmax(self.arch_reduce_parameters, dim=-1).cpu().numpy())
        return {'normal': gene_normal, 'normal_concat': list(range(2 + self._steps - self._multiplier, self._steps + 2)), 'reduce': gene_reduce, 'reduce_concat': list(range(2 + self._steps - self._multiplier, self._steps + 2))}

    def forward(self, inputs):
        normal_w = nn.functional.softmax(self.arch_normal_parameters, dim=1)
        reduce_w = nn.functional.softmax(self.arch_reduce_parameters, dim=1)
        s0 = s1 = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                ww = reduce_w
            else:
                ww = normal_w
            s0, s1 = s1, cell.forward_darts(s0, s1, ww)
        out = self.lastact(s1)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits


class TinyNetworkENAS(nn.Module):

    def __init__(self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats):
        super(TinyNetworkENAS, self).__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C))
        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.sampled_arch = None

    def update_arch(self, _arch):
        if _arch is None:
            self.sampled_arch = None
        elif isinstance(_arch, Structure):
            self.sampled_arch = _arch
        elif isinstance(_arch, (list, tuple)):
            genotypes = []
            for i in range(1, self.max_nodes):
                xlist = []
                for j in range(i):
                    node_str = '{:}<-{:}'.format(i, j)
                    op_index = _arch[self.edge2index[node_str]]
                    op_name = self.op_names[op_index]
                    xlist.append((op_name, j))
                genotypes.append(tuple(xlist))
            self.sampled_arch = Structure(genotypes)
        else:
            raise ValueError('invalid type of input architecture : {:}'.format(_arch))
        return self.sampled_arch

    def create_controller(self):
        return Controller(len(self.edge2index), len(self.op_names))

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return '{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                feature = cell.forward_dynamic(feature, self.sampled_arch)
            else:
                feature = cell(feature)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits


class Controller(nn.Module):

    def __init__(self, num_edge, num_ops, lstm_size=32, lstm_num_layers=2, tanh_constant=2.5, temperature=5.0):
        super(Controller, self).__init__()
        self.num_edge = num_edge
        self.num_ops = num_ops
        self.lstm_size = lstm_size
        self.lstm_N = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature
        self.register_parameter('input_vars', nn.Parameter(torch.Tensor(1, 1, lstm_size)))
        self.w_lstm = nn.LSTM(input_size=self.lstm_size, hidden_size=self.lstm_size, num_layers=self.lstm_N)
        self.w_embd = nn.Embedding(self.num_ops, self.lstm_size)
        self.w_pred = nn.Linear(self.lstm_size, self.num_ops)
        nn.init.uniform_(self.input_vars, -0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_hh_l0, -0.1, 0.1)
        nn.init.uniform_(self.w_lstm.weight_ih_l0, -0.1, 0.1)
        nn.init.uniform_(self.w_embd.weight, -0.1, 0.1)
        nn.init.uniform_(self.w_pred.weight, -0.1, 0.1)

    def forward(self):
        inputs, h0 = self.input_vars, None
        log_probs, entropys, sampled_arch = [], [], []
        for iedge in range(self.num_edge):
            outputs, h0 = self.w_lstm(inputs, h0)
            logits = self.w_pred(outputs)
            logits = logits / self.temperature
            logits = self.tanh_constant * torch.tanh(logits)
            op_distribution = Categorical(logits=logits)
            op_index = op_distribution.sample()
            sampled_arch.append(op_index.item())
            op_log_prob = op_distribution.log_prob(op_index)
            log_probs.append(op_log_prob.view(-1))
            op_entropy = op_distribution.entropy()
            entropys.append(op_entropy.view(-1))
            inputs = self.w_embd(op_index)
        return torch.sum(torch.cat(log_probs)), torch.sum(torch.cat(entropys)), sampled_arch


class TinyNetworkGDAS(nn.Module):

    def __init__(self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats):
        super(TinyNetworkGDAS, self).__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C))
        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch_parameters = nn.Parameter(0.001 * torch.randn(num_edge, len(search_space)))
        self.tau = 10

    def get_weights(self):
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(self.global_pooling.parameters())
        xlist += list(self.classifier.parameters())
        return xlist

    def set_tau(self, tau):
        self.tau = tau

    def get_tau(self):
        return self.tau

    def get_alphas(self):
        return [self.arch_parameters]

    def show_alphas(self):
        with torch.no_grad():
            return 'arch-parameters :\n{:}'.format(nn.functional.softmax(self.arch_parameters, dim=-1).cpu())

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return '{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__)

    def genotype(self):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    weights = self.arch_parameters[self.edge2index[node_str]]
                    op_name = self.op_names[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def forward(self, inputs):
        while True:
            gumbels = -torch.empty_like(self.arch_parameters).exponential_().log()
            logits = (self.arch_parameters.log_softmax(dim=1) + gumbels) / self.tau
            probs = nn.functional.softmax(logits, dim=1)
            index = probs.max(-1, keepdim=True)[1]
            one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            hardwts = one_h - probs.detach() + probs
            if torch.isinf(gumbels).any() or torch.isinf(probs).any() or torch.isnan(probs).any():
                continue
            else:
                break
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                feature = cell.forward_gdas(feature, hardwts, index)
            else:
                feature = cell(feature)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits


class NASNetworkGDAS(nn.Module):

    def __init__(self, C, N, steps, multiplier, stem_multiplier, num_classes, search_space, affine, track_running_stats):
        super(NASNetworkGDAS, self).__init__()
        self._C = C
        self._layerN = N
        self._steps = steps
        self._multiplier = multiplier
        self.stem = nn.Sequential(nn.Conv2d(3, C * stem_multiplier, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C * stem_multiplier))
        layer_channels = [C] * N + [C * 2] + [C * 2] * (N - 1) + [C * 4] + [C * 4] * (N - 1)
        layer_reductions = [False] * N + [True] + [False] * (N - 1) + [True] + [False] * (N - 1)
        num_edge, edge2index = None, None
        C_prev_prev, C_prev, C_curr, reduction_prev = C * stem_multiplier, C * stem_multiplier, C, False
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            cell = SearchCell(search_space, steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, affine, track_running_stats)
            if num_edge is None:
                num_edge, edge2index = cell.num_edges, cell.edge2index
            else:
                assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev_prev, C_prev, reduction_prev = C_prev, multiplier * C_curr, reduction
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch_normal_parameters = nn.Parameter(0.001 * torch.randn(num_edge, len(search_space)))
        self.arch_reduce_parameters = nn.Parameter(0.001 * torch.randn(num_edge, len(search_space)))
        self.tau = 10

    def get_weights(self):
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(self.global_pooling.parameters())
        xlist += list(self.classifier.parameters())
        return xlist

    def set_tau(self, tau):
        self.tau = tau

    def get_tau(self):
        return self.tau

    def get_alphas(self):
        return [self.arch_normal_parameters, self.arch_reduce_parameters]

    def show_alphas(self):
        with torch.no_grad():
            A = 'arch-normal-parameters :\n{:}'.format(nn.functional.softmax(self.arch_normal_parameters, dim=-1).cpu())
            B = 'arch-reduce-parameters :\n{:}'.format(nn.functional.softmax(self.arch_reduce_parameters, dim=-1).cpu())
        return '{:}\n{:}'.format(A, B)

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return '{name}(C={_C}, N={_layerN}, steps={_steps}, multiplier={_multiplier}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__)

    def genotype(self):

        def _parse(weights):
            gene = []
            for i in range(self._steps):
                edges = []
                for j in range(2 + i):
                    node_str = '{:}<-{:}'.format(i, j)
                    ws = weights[self.edge2index[node_str]]
                    for k, op_name in enumerate(self.op_names):
                        if op_name == 'none':
                            continue
                        edges.append((op_name, j, ws[k]))
                edges = sorted(edges, key=lambda x: -x[-1])
                selected_edges = edges[:2]
                gene.append(tuple(selected_edges))
            return gene
        with torch.no_grad():
            gene_normal = _parse(torch.softmax(self.arch_normal_parameters, dim=-1).cpu().numpy())
            gene_reduce = _parse(torch.softmax(self.arch_reduce_parameters, dim=-1).cpu().numpy())
        return {'normal': gene_normal, 'normal_concat': list(range(2 + self._steps - self._multiplier, self._steps + 2)), 'reduce': gene_reduce, 'reduce_concat': list(range(2 + self._steps - self._multiplier, self._steps + 2))}

    def forward(self, inputs):

        def get_gumbel_prob(xins):
            while True:
                gumbels = -torch.empty_like(xins).exponential_().log()
                logits = (xins.log_softmax(dim=1) + gumbels) / self.tau
                probs = nn.functional.softmax(logits, dim=1)
                index = probs.max(-1, keepdim=True)[1]
                one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)
                hardwts = one_h - probs.detach() + probs
                if torch.isinf(gumbels).any() or torch.isinf(probs).any() or torch.isnan(probs).any():
                    continue
                else:
                    break
            return hardwts, index
        normal_hardwts, normal_index = get_gumbel_prob(self.arch_normal_parameters)
        reduce_hardwts, reduce_index = get_gumbel_prob(self.arch_reduce_parameters)
        s0 = s1 = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                hardwts, index = reduce_hardwts, reduce_index
            else:
                hardwts, index = normal_hardwts, normal_index
            s0, s1 = s1, cell.forward_gdas(s0, s1, hardwts, index)
        out = self.lastact(s1)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits


class TinyNetworkRANDOM(nn.Module):

    def __init__(self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats):
        super(TinyNetworkRANDOM, self).__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C))
        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch_cache = None

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return '{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__)

    def random_genotype(self, set_cache):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                op_name = random.choice(self.op_names)
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        arch = Structure(genotypes)
        if set_cache:
            self.arch_cache = arch
        return arch

    def forward(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                feature = cell.forward_dynamic(feature, self.arch_cache)
            else:
                feature = cell(feature)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits


class TinyNetworkSETN(nn.Module):

    def __init__(self, C, N, max_nodes, num_classes, search_space, affine, track_running_stats):
        super(TinyNetworkSETN, self).__init__()
        self._C = C
        self._layerN = N
        self.max_nodes = max_nodes
        self.stem = nn.Sequential(nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C))
        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
        C_prev, num_edge, edge2index = C, None, None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                cell = SearchCell(C_prev, C_curr, 1, max_nodes, search_space, affine, track_running_stats)
                if num_edge is None:
                    num_edge, edge2index = cell.num_edges, cell.edge2index
                else:
                    assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev = cell.out_dim
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch_parameters = nn.Parameter(0.001 * torch.randn(num_edge, len(search_space)))
        self.mode = 'urs'
        self.dynamic_cell = None

    def set_cal_mode(self, mode, dynamic_cell=None):
        assert mode in ['urs', 'joint', 'select', 'dynamic']
        self.mode = mode
        if mode == 'dynamic':
            self.dynamic_cell = deepcopy(dynamic_cell)
        else:
            self.dynamic_cell = None

    def get_cal_mode(self):
        return self.mode

    def get_weights(self):
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(self.global_pooling.parameters())
        xlist += list(self.classifier.parameters())
        return xlist

    def get_alphas(self):
        return [self.arch_parameters]

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return '{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__)

    def genotype(self):
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                with torch.no_grad():
                    weights = self.arch_parameters[self.edge2index[node_str]]
                    op_name = self.op_names[weights.argmax().item()]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def dync_genotype(self, use_random=False):
        genotypes = []
        with torch.no_grad():
            alphas_cpu = nn.functional.softmax(self.arch_parameters, dim=-1)
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                if use_random:
                    op_name = random.choice(self.op_names)
                else:
                    weights = alphas_cpu[self.edge2index[node_str]]
                    op_index = torch.multinomial(weights, 1).item()
                    op_name = self.op_names[op_index]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def get_log_prob(self, arch):
        with torch.no_grad():
            logits = nn.functional.log_softmax(self.arch_parameters, dim=-1)
        select_logits = []
        for i, node_info in enumerate(arch.nodes):
            for op, xin in node_info:
                node_str = '{:}<-{:}'.format(i + 1, xin)
                op_index = self.op_names.index(op)
                select_logits.append(logits[self.edge2index[node_str], op_index])
        return sum(select_logits).item()

    def return_topK(self, K):
        archs = Structure.gen_all(self.op_names, self.max_nodes, False)
        pairs = [(self.get_log_prob(arch), arch) for arch in archs]
        if K < 0 or K >= len(archs):
            K = len(archs)
        sorted_pairs = sorted(pairs, key=lambda x: -x[0])
        return_pairs = [sorted_pairs[_][1] for _ in range(K)]
        return return_pairs

    def forward(self, inputs):
        alphas = nn.functional.softmax(self.arch_parameters, dim=-1)
        with torch.no_grad():
            alphas_cpu = alphas.detach().cpu()
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            if isinstance(cell, SearchCell):
                if self.mode == 'urs':
                    feature = cell.forward_urs(feature)
                elif self.mode == 'select':
                    feature = cell.forward_select(feature, alphas_cpu)
                elif self.mode == 'joint':
                    feature = cell.forward_joint(feature, alphas)
                elif self.mode == 'dynamic':
                    feature = cell.forward_dynamic(feature, self.dynamic_cell)
                else:
                    raise ValueError('invalid mode={:}'.format(self.mode))
            else:
                feature = cell(feature)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits


class NASNetworkSETN(nn.Module):

    def __init__(self, C: int, N: int, steps: int, multiplier: int, stem_multiplier: int, num_classes: int, search_space: List[Text], affine: bool, track_running_stats: bool):
        super(NASNetworkSETN, self).__init__()
        self._C = C
        self._layerN = N
        self._steps = steps
        self._multiplier = multiplier
        self.stem = nn.Sequential(nn.Conv2d(3, C * stem_multiplier, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C * stem_multiplier))
        layer_channels = [C] * N + [C * 2] + [C * 2] * (N - 1) + [C * 4] + [C * 4] * (N - 1)
        layer_reductions = [False] * N + [True] + [False] * (N - 1) + [True] + [False] * (N - 1)
        num_edge, edge2index = None, None
        C_prev_prev, C_prev, C_curr, reduction_prev = C * stem_multiplier, C * stem_multiplier, C, False
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            cell = SearchCell(search_space, steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, affine, track_running_stats)
            if num_edge is None:
                num_edge, edge2index = cell.num_edges, cell.edge2index
            else:
                assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
            self.cells.append(cell)
            C_prev_prev, C_prev, reduction_prev = C_prev, multiplier * C_curr, reduction
        self.op_names = deepcopy(search_space)
        self._Layer = len(self.cells)
        self.edge2index = edge2index
        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.arch_normal_parameters = nn.Parameter(0.001 * torch.randn(num_edge, len(search_space)))
        self.arch_reduce_parameters = nn.Parameter(0.001 * torch.randn(num_edge, len(search_space)))
        self.mode = 'urs'
        self.dynamic_cell = None

    def set_cal_mode(self, mode, dynamic_cell=None):
        assert mode in ['urs', 'joint', 'select', 'dynamic']
        self.mode = mode
        if mode == 'dynamic':
            self.dynamic_cell = deepcopy(dynamic_cell)
        else:
            self.dynamic_cell = None

    def get_weights(self):
        xlist = list(self.stem.parameters()) + list(self.cells.parameters())
        xlist += list(self.lastact.parameters()) + list(self.global_pooling.parameters())
        xlist += list(self.classifier.parameters())
        return xlist

    def get_alphas(self):
        return [self.arch_normal_parameters, self.arch_reduce_parameters]

    def show_alphas(self):
        with torch.no_grad():
            A = 'arch-normal-parameters :\n{:}'.format(nn.functional.softmax(self.arch_normal_parameters, dim=-1).cpu())
            B = 'arch-reduce-parameters :\n{:}'.format(nn.functional.softmax(self.arch_reduce_parameters, dim=-1).cpu())
        return '{:}\n{:}'.format(A, B)

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return '{name}(C={_C}, N={_layerN}, steps={_steps}, multiplier={_multiplier}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__)

    def dync_genotype(self, use_random=False):
        genotypes = []
        with torch.no_grad():
            alphas_cpu = nn.functional.softmax(self.arch_parameters, dim=-1)
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = '{:}<-{:}'.format(i, j)
                if use_random:
                    op_name = random.choice(self.op_names)
                else:
                    weights = alphas_cpu[self.edge2index[node_str]]
                    op_index = torch.multinomial(weights, 1).item()
                    op_name = self.op_names[op_index]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes)

    def genotype(self):

        def _parse(weights):
            gene = []
            for i in range(self._steps):
                edges = []
                for j in range(2 + i):
                    node_str = '{:}<-{:}'.format(i, j)
                    ws = weights[self.edge2index[node_str]]
                    for k, op_name in enumerate(self.op_names):
                        if op_name == 'none':
                            continue
                        edges.append((op_name, j, ws[k]))
                edges = sorted(edges, key=lambda x: -x[-1])
                selected_edges = edges[:2]
                gene.append(tuple(selected_edges))
            return gene
        with torch.no_grad():
            gene_normal = _parse(torch.softmax(self.arch_normal_parameters, dim=-1).cpu().numpy())
            gene_reduce = _parse(torch.softmax(self.arch_reduce_parameters, dim=-1).cpu().numpy())
        return {'normal': gene_normal, 'normal_concat': list(range(2 + self._steps - self._multiplier, self._steps + 2)), 'reduce': gene_reduce, 'reduce_concat': list(range(2 + self._steps - self._multiplier, self._steps + 2))}

    def forward(self, inputs):
        normal_hardwts = nn.functional.softmax(self.arch_normal_parameters, dim=-1)
        reduce_hardwts = nn.functional.softmax(self.arch_reduce_parameters, dim=-1)
        s0 = s1 = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            raise NotImplementedError
            if cell.reduction:
                hardwts, index = reduce_hardwts, reduce_index
            else:
                hardwts, index = normal_hardwts, normal_index
            s0, s1 = s1, cell.forward_gdas(s0, s1, hardwts, index)
        out = self.lastact(s1)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits


class ConvBNReLU(nn.Module):

    def __init__(self, nIn, nOut, kernel, stride, padding, bias, has_avg, has_bn, has_relu):
        super(ConvBNReLU, self).__init__()
        if has_avg:
            self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.avg = None
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride, padding=padding, dilation=1, groups=1, bias=bias)
        if has_bn:
            self.bn = nn.BatchNorm2d(nOut)
        else:
            self.bn = None
        if has_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, inputs):
        if self.avg:
            out = self.avg(inputs)
        else:
            out = inputs
        conv = self.conv(out)
        if self.bn:
            out = self.bn(conv)
        else:
            out = conv
        if self.relu:
            out = self.relu(out)
        else:
            out = out
        return out


class ResNetBasicblock(nn.Module):
    num_conv = 2
    expansion = 1

    def __init__(self, iCs, stride):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        assert isinstance(iCs, tuple) or isinstance(iCs, list), 'invalid type of iCs : {:}'.format(iCs)
        assert len(iCs) == 3, 'invalid lengths of iCs : {:}'.format(iCs)
        self.conv_a = ConvBNReLU(iCs[0], iCs[1], 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_b = ConvBNReLU(iCs[1], iCs[2], 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=False)
        residual_in = iCs[0]
        if stride == 2:
            self.downsample = ConvBNReLU(iCs[0], iCs[2], 1, 1, 0, False, has_avg=True, has_bn=False, has_relu=False)
            residual_in = iCs[2]
        elif iCs[0] != iCs[2]:
            self.downsample = ConvBNReLU(iCs[0], iCs[2], 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        else:
            self.downsample = None
        self.out_dim = iCs[2]

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = residual + basicblock
        return F.relu(out, inplace=True)


class ResNetBottleneck(nn.Module):
    expansion = 4
    num_conv = 3

    def __init__(self, iCs, stride):
        super(ResNetBottleneck, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        assert isinstance(iCs, tuple) or isinstance(iCs, list), 'invalid type of iCs : {:}'.format(iCs)
        assert len(iCs) == 4, 'invalid lengths of iCs : {:}'.format(iCs)
        self.conv_1x1 = ConvBNReLU(iCs[0], iCs[1], 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_3x3 = ConvBNReLU(iCs[1], iCs[2], 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_1x4 = ConvBNReLU(iCs[2], iCs[3], 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        residual_in = iCs[0]
        if stride == 2:
            self.downsample = ConvBNReLU(iCs[0], iCs[3], 1, 1, 0, False, has_avg=True, has_bn=False, has_relu=False)
            residual_in = iCs[3]
        elif iCs[0] != iCs[3]:
            self.downsample = ConvBNReLU(iCs[0], iCs[3], 1, 1, 0, False, has_avg=False, has_bn=False, has_relu=False)
            residual_in = iCs[3]
        else:
            self.downsample = None
        self.out_dim = iCs[3]

    def forward(self, inputs):
        bottleneck = self.conv_1x1(inputs)
        bottleneck = self.conv_3x3(bottleneck)
        bottleneck = self.conv_1x4(bottleneck)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = residual + bottleneck
        return F.relu(out, inplace=True)


class InferCifarResNet(nn.Module):

    def __init__(self, block_name, depth, xblocks, xchannels, num_classes, zero_init_residual):
        super(InferCifarResNet, self).__init__()
        if block_name == 'ResNetBasicblock':
            block = ResNetBasicblock
            assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
            layer_blocks = (depth - 2) // 6
        elif block_name == 'ResNetBottleneck':
            block = ResNetBottleneck
            assert (depth - 2) % 9 == 0, 'depth should be one of 164'
            layer_blocks = (depth - 2) // 9
        else:
            raise ValueError('invalid block : {:}'.format(block_name))
        assert len(xblocks) == 3, 'invalid xblocks : {:}'.format(xblocks)
        self.message = 'InferWidthCifarResNet : Depth : {:} , Layers for each block : {:}'.format(depth, layer_blocks)
        self.num_classes = num_classes
        self.xchannels = xchannels
        self.layers = nn.ModuleList([ConvBNReLU(xchannels[0], xchannels[1], 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=True)])
        last_channel_idx = 1
        for stage in range(3):
            for iL in range(layer_blocks):
                num_conv = block.num_conv
                iCs = self.xchannels[last_channel_idx:last_channel_idx + num_conv + 1]
                stride = 2 if stage > 0 and iL == 0 else 1
                module = block(iCs, stride)
                last_channel_idx += num_conv
                self.xchannels[last_channel_idx] = module.out_dim
                self.layers.append(module)
                self.message += '\nstage={:}, ilayer={:02d}/{:02d}, block={:03d}, iCs={:}, oC={:3d}, stride={:}'.format(stage, iL, layer_blocks, len(self.layers) - 1, iCs, module.out_dim, stride)
                if iL + 1 == xblocks[stage]:
                    out_channel = module.out_dim
                    for iiL in range(iL + 1, layer_blocks):
                        last_channel_idx += num_conv
                    self.xchannels[last_channel_idx] = module.out_dim
                    break
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(self.xchannels[-1], num_classes)
        self.apply(initialize_resnet)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNetBasicblock):
                    nn.init.constant_(m.conv_b.bn.weight, 0)
                elif isinstance(m, ResNetBottleneck):
                    nn.init.constant_(m.conv_1x4.bn.weight, 0)

    def get_message(self):
        return self.message

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return features, logits


class ConvBNReLU(nn.Module):

    def __init__(self, nIn, nOut, kernel, stride, padding, bias, has_avg, has_bn, has_relu):
        super(ConvBNReLU, self).__init__()
        if has_avg:
            self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.avg = None
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride, padding=padding, dilation=1, groups=1, bias=bias)
        if has_bn:
            self.bn = nn.BatchNorm2d(nOut)
        else:
            self.bn = None
        if has_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, inputs):
        if self.avg:
            out = self.avg(inputs)
        else:
            out = inputs
        conv = self.conv(out)
        if self.bn:
            out = self.bn(conv)
        else:
            out = conv
        if self.relu:
            out = self.relu(out)
        else:
            out = out
        return out


class ResNetBasicblock(nn.Module):
    num_conv = 2
    expansion = 1

    def __init__(self, inplanes, planes, stride):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ConvBNReLU(inplanes, planes, 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_b = ConvBNReLU(planes, planes, 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=False)
        if stride == 2:
            self.downsample = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=True, has_bn=False, has_relu=False)
        elif inplanes != planes:
            self.downsample = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        else:
            self.downsample = None
        self.out_dim = planes

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = residual + basicblock
        return F.relu(out, inplace=True)


class ResNetBottleneck(nn.Module):
    expansion = 4
    num_conv = 3

    def __init__(self, inplanes, planes, stride):
        super(ResNetBottleneck, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_1x1 = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_3x3 = ConvBNReLU(planes, planes, 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_1x4 = ConvBNReLU(planes, planes * self.expansion, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        if stride == 2:
            self.downsample = ConvBNReLU(inplanes, planes * self.expansion, 1, 1, 0, False, has_avg=True, has_bn=False, has_relu=False)
        elif inplanes != planes * self.expansion:
            self.downsample = ConvBNReLU(inplanes, planes * self.expansion, 1, 1, 0, False, has_avg=False, has_bn=False, has_relu=False)
        else:
            self.downsample = None
        self.out_dim = planes * self.expansion

    def forward(self, inputs):
        bottleneck = self.conv_1x1(inputs)
        bottleneck = self.conv_3x3(bottleneck)
        bottleneck = self.conv_1x4(bottleneck)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = residual + bottleneck
        return F.relu(out, inplace=True)


class InferDepthCifarResNet(nn.Module):

    def __init__(self, block_name, depth, xblocks, num_classes, zero_init_residual):
        super(InferDepthCifarResNet, self).__init__()
        if block_name == 'ResNetBasicblock':
            block = ResNetBasicblock
            assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
            layer_blocks = (depth - 2) // 6
        elif block_name == 'ResNetBottleneck':
            block = ResNetBottleneck
            assert (depth - 2) % 9 == 0, 'depth should be one of 164'
            layer_blocks = (depth - 2) // 9
        else:
            raise ValueError('invalid block : {:}'.format(block_name))
        assert len(xblocks) == 3, 'invalid xblocks : {:}'.format(xblocks)
        self.message = 'InferWidthCifarResNet : Depth : {:} , Layers for each block : {:}'.format(depth, layer_blocks)
        self.num_classes = num_classes
        self.layers = nn.ModuleList([ConvBNReLU(3, 16, 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=True)])
        self.channels = [16]
        for stage in range(3):
            for iL in range(layer_blocks):
                iC = self.channels[-1]
                planes = 16 * 2 ** stage
                stride = 2 if stage > 0 and iL == 0 else 1
                module = block(iC, planes, stride)
                self.channels.append(module.out_dim)
                self.layers.append(module)
                self.message += '\nstage={:}, ilayer={:02d}/{:02d}, block={:03d}, iC={:}, oC={:3d}, stride={:}'.format(stage, iL, layer_blocks, len(self.layers) - 1, planes, module.out_dim, stride)
                if iL + 1 == xblocks[stage]:
                    break
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(self.channels[-1], num_classes)
        self.apply(initialize_resnet)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNetBasicblock):
                    nn.init.constant_(m.conv_b.bn.weight, 0)
                elif isinstance(m, ResNetBottleneck):
                    nn.init.constant_(m.conv_1x4.bn.weight, 0)

    def get_message(self):
        return self.message

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return features, logits


class ConvBNReLU(nn.Module):

    def __init__(self, nIn, nOut, kernel, stride, padding, bias, has_avg, has_bn, has_relu):
        super(ConvBNReLU, self).__init__()
        if has_avg:
            self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.avg = None
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride, padding=padding, dilation=1, groups=1, bias=bias)
        if has_bn:
            self.bn = nn.BatchNorm2d(nOut)
        else:
            self.bn = None
        if has_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, inputs):
        if self.avg:
            out = self.avg(inputs)
        else:
            out = inputs
        conv = self.conv(out)
        if self.bn:
            out = self.bn(conv)
        else:
            out = conv
        if self.relu:
            out = self.relu(out)
        else:
            out = out
        return out


class ResNetBasicblock(nn.Module):
    num_conv = 2
    expansion = 1

    def __init__(self, iCs, stride):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        assert isinstance(iCs, tuple) or isinstance(iCs, list), 'invalid type of iCs : {:}'.format(iCs)
        assert len(iCs) == 3, 'invalid lengths of iCs : {:}'.format(iCs)
        self.conv_a = ConvBNReLU(iCs[0], iCs[1], 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_b = ConvBNReLU(iCs[1], iCs[2], 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=False)
        residual_in = iCs[0]
        if stride == 2:
            self.downsample = ConvBNReLU(iCs[0], iCs[2], 1, 1, 0, False, has_avg=True, has_bn=False, has_relu=False)
            residual_in = iCs[2]
        elif iCs[0] != iCs[2]:
            self.downsample = ConvBNReLU(iCs[0], iCs[2], 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        else:
            self.downsample = None
        self.out_dim = iCs[2]

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = residual + basicblock
        return F.relu(out, inplace=True)


class ResNetBottleneck(nn.Module):
    expansion = 4
    num_conv = 3

    def __init__(self, iCs, stride):
        super(ResNetBottleneck, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        assert isinstance(iCs, tuple) or isinstance(iCs, list), 'invalid type of iCs : {:}'.format(iCs)
        assert len(iCs) == 4, 'invalid lengths of iCs : {:}'.format(iCs)
        self.conv_1x1 = ConvBNReLU(iCs[0], iCs[1], 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_3x3 = ConvBNReLU(iCs[1], iCs[2], 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_1x4 = ConvBNReLU(iCs[2], iCs[3], 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        residual_in = iCs[0]
        if stride == 2:
            self.downsample = ConvBNReLU(iCs[0], iCs[3], 1, 1, 0, False, has_avg=True, has_bn=False, has_relu=False)
            residual_in = iCs[3]
        elif iCs[0] != iCs[3]:
            self.downsample = ConvBNReLU(iCs[0], iCs[3], 1, 1, 0, False, has_avg=False, has_bn=False, has_relu=False)
            residual_in = iCs[3]
        else:
            self.downsample = None
        self.out_dim = iCs[3]

    def forward(self, inputs):
        bottleneck = self.conv_1x1(inputs)
        bottleneck = self.conv_3x3(bottleneck)
        bottleneck = self.conv_1x4(bottleneck)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = residual + bottleneck
        return F.relu(out, inplace=True)


class InferWidthCifarResNet(nn.Module):

    def __init__(self, block_name, depth, xchannels, num_classes, zero_init_residual):
        super(InferWidthCifarResNet, self).__init__()
        if block_name == 'ResNetBasicblock':
            block = ResNetBasicblock
            assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
            layer_blocks = (depth - 2) // 6
        elif block_name == 'ResNetBottleneck':
            block = ResNetBottleneck
            assert (depth - 2) % 9 == 0, 'depth should be one of 164'
            layer_blocks = (depth - 2) // 9
        else:
            raise ValueError('invalid block : {:}'.format(block_name))
        self.message = 'InferWidthCifarResNet : Depth : {:} , Layers for each block : {:}'.format(depth, layer_blocks)
        self.num_classes = num_classes
        self.xchannels = xchannels
        self.layers = nn.ModuleList([ConvBNReLU(xchannels[0], xchannels[1], 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=True)])
        last_channel_idx = 1
        for stage in range(3):
            for iL in range(layer_blocks):
                num_conv = block.num_conv
                iCs = self.xchannels[last_channel_idx:last_channel_idx + num_conv + 1]
                stride = 2 if stage > 0 and iL == 0 else 1
                module = block(iCs, stride)
                last_channel_idx += num_conv
                self.xchannels[last_channel_idx] = module.out_dim
                self.layers.append(module)
                self.message += '\nstage={:}, ilayer={:02d}/{:02d}, block={:03d}, iCs={:}, oC={:3d}, stride={:}'.format(stage, iL, layer_blocks, len(self.layers) - 1, iCs, module.out_dim, stride)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(self.xchannels[-1], num_classes)
        self.apply(initialize_resnet)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNetBasicblock):
                    nn.init.constant_(m.conv_b.bn.weight, 0)
                elif isinstance(m, ResNetBottleneck):
                    nn.init.constant_(m.conv_1x4.bn.weight, 0)

    def get_message(self):
        return self.message

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return features, logits


class ConvBNReLU(nn.Module):
    num_conv = 1

    def __init__(self, nIn, nOut, kernel, stride, padding, bias, has_avg, has_bn, has_relu):
        super(ConvBNReLU, self).__init__()
        if has_avg:
            self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.avg = None
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride, padding=padding, dilation=1, groups=1, bias=bias)
        if has_bn:
            self.bn = nn.BatchNorm2d(nOut)
        else:
            self.bn = None
        if has_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, inputs):
        if self.avg:
            out = self.avg(inputs)
        else:
            out = inputs
        conv = self.conv(out)
        if self.bn:
            out = self.bn(conv)
        else:
            out = conv
        if self.relu:
            out = self.relu(out)
        else:
            out = out
        return out


class ResNetBasicblock(nn.Module):
    num_conv = 2
    expansion = 1

    def __init__(self, iCs, stride):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        assert isinstance(iCs, tuple) or isinstance(iCs, list), 'invalid type of iCs : {:}'.format(iCs)
        assert len(iCs) == 3, 'invalid lengths of iCs : {:}'.format(iCs)
        self.conv_a = ConvBNReLU(iCs[0], iCs[1], 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_b = ConvBNReLU(iCs[1], iCs[2], 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=False)
        residual_in = iCs[0]
        if stride == 2:
            self.downsample = ConvBNReLU(iCs[0], iCs[2], 1, 1, 0, False, has_avg=True, has_bn=True, has_relu=False)
            residual_in = iCs[2]
        elif iCs[0] != iCs[2]:
            self.downsample = ConvBNReLU(iCs[0], iCs[2], 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        else:
            self.downsample = None
        self.out_dim = iCs[2]

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = residual + basicblock
        return F.relu(out, inplace=True)


class ResNetBottleneck(nn.Module):
    expansion = 4
    num_conv = 3

    def __init__(self, iCs, stride):
        super(ResNetBottleneck, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        assert isinstance(iCs, tuple) or isinstance(iCs, list), 'invalid type of iCs : {:}'.format(iCs)
        assert len(iCs) == 4, 'invalid lengths of iCs : {:}'.format(iCs)
        self.conv_1x1 = ConvBNReLU(iCs[0], iCs[1], 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_3x3 = ConvBNReLU(iCs[1], iCs[2], 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_1x4 = ConvBNReLU(iCs[2], iCs[3], 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        residual_in = iCs[0]
        if stride == 2:
            self.downsample = ConvBNReLU(iCs[0], iCs[3], 1, 1, 0, False, has_avg=True, has_bn=True, has_relu=False)
            residual_in = iCs[3]
        elif iCs[0] != iCs[3]:
            self.downsample = ConvBNReLU(iCs[0], iCs[3], 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
            residual_in = iCs[3]
        else:
            self.downsample = None
        self.out_dim = iCs[3]

    def forward(self, inputs):
        bottleneck = self.conv_1x1(inputs)
        bottleneck = self.conv_3x3(bottleneck)
        bottleneck = self.conv_1x4(bottleneck)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = residual + bottleneck
        return F.relu(out, inplace=True)


class InferImagenetResNet(nn.Module):

    def __init__(self, block_name, layers, xblocks, xchannels, deep_stem, num_classes, zero_init_residual):
        super(InferImagenetResNet, self).__init__()
        if block_name == 'BasicBlock':
            block = ResNetBasicblock
        elif block_name == 'Bottleneck':
            block = ResNetBottleneck
        else:
            raise ValueError('invalid block : {:}'.format(block_name))
        assert len(xblocks) == len(layers), 'invalid layers : {:} vs xblocks : {:}'.format(layers, xblocks)
        self.message = 'InferImagenetResNet : Depth : {:} -> {:}, Layers for each block : {:}'.format(sum(layers) * block.num_conv, sum(xblocks) * block.num_conv, xblocks)
        self.num_classes = num_classes
        self.xchannels = xchannels
        if not deep_stem:
            self.layers = nn.ModuleList([ConvBNReLU(xchannels[0], xchannels[1], 7, 2, 3, False, has_avg=False, has_bn=True, has_relu=True)])
            last_channel_idx = 1
        else:
            self.layers = nn.ModuleList([ConvBNReLU(xchannels[0], xchannels[1], 3, 2, 1, False, has_avg=False, has_bn=True, has_relu=True), ConvBNReLU(xchannels[1], xchannels[2], 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=True)])
            last_channel_idx = 2
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for stage, layer_blocks in enumerate(layers):
            for iL in range(layer_blocks):
                num_conv = block.num_conv
                iCs = self.xchannels[last_channel_idx:last_channel_idx + num_conv + 1]
                stride = 2 if stage > 0 and iL == 0 else 1
                module = block(iCs, stride)
                last_channel_idx += num_conv
                self.xchannels[last_channel_idx] = module.out_dim
                self.layers.append(module)
                self.message += '\nstage={:}, ilayer={:02d}/{:02d}, block={:03d}, iCs={:}, oC={:3d}, stride={:}'.format(stage, iL, layer_blocks, len(self.layers) - 1, iCs, module.out_dim, stride)
                if iL + 1 == xblocks[stage]:
                    out_channel = module.out_dim
                    for iiL in range(iL + 1, layer_blocks):
                        last_channel_idx += num_conv
                    self.xchannels[last_channel_idx] = module.out_dim
                    break
        assert last_channel_idx + 1 == len(self.xchannels), '{:} vs {:}'.format(last_channel_idx, len(self.xchannels))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.xchannels[-1], num_classes)
        self.apply(initialize_resnet)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNetBasicblock):
                    nn.init.constant_(m.conv_b.bn.weight, 0)
                elif isinstance(m, ResNetBottleneck):
                    nn.init.constant_(m.conv_1x4.bn.weight, 0)

    def get_message(self):
        return self.message

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return features, logits


class ConvBNReLU(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, groups, has_bn=True, has_relu=True):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False)
        if has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        else:
            self.bn = None
        if has_relu:
            self.relu = nn.ReLU6(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        out = self.conv(x)
        if self.bn:
            out = self.bn(out)
        if self.relu:
            out = self.relu(out)
        return out


class InvertedResidual(nn.Module):

    def __init__(self, channels, stride, expand_ratio, additive):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], 'invalid stride : {:}'.format(stride)
        assert len(channels) in [2, 3], 'invalid channels : {:}'.format(channels)
        if len(channels) == 2:
            layers = []
        else:
            layers = [ConvBNReLU(channels[0], channels[1], 1, 1, 1)]
        layers.extend([ConvBNReLU(channels[-2], channels[-2], 3, stride, channels[-2]), ConvBNReLU(channels[-2], channels[-1], 1, 1, 1, True, False)])
        self.conv = nn.Sequential(*layers)
        self.additive = additive
        if self.additive and channels[0] != channels[-1]:
            self.shortcut = ConvBNReLU(channels[0], channels[-1], 1, 1, 1, True, False)
        else:
            self.shortcut = None
        self.out_dim = channels[-1]

    def forward(self, x):
        out = self.conv(x)
        if self.shortcut:
            return out + self.shortcut(x)
        else:
            return out


def parse_channel_info(xstring):
    blocks = xstring.split(' ')
    blocks = [x.split('-') for x in blocks]
    blocks = [[int(_) for _ in x] for x in blocks]
    return blocks


class InferMobileNetV2(nn.Module):

    def __init__(self, num_classes, xchannels, xblocks, dropout):
        super(InferMobileNetV2, self).__init__()
        block = InvertedResidual
        inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        assert len(inverted_residual_setting) == len(xblocks), 'invalid number of layers : {:} vs {:}'.format(len(inverted_residual_setting), len(xblocks))
        for block_num, ir_setting in zip(xblocks, inverted_residual_setting):
            assert block_num <= ir_setting[2], '{:} vs {:}'.format(block_num, ir_setting)
        xchannels = parse_channel_info(xchannels)
        self.xchannels = xchannels
        self.message = 'InferMobileNetV2 : xblocks={:}'.format(xblocks)
        features = [ConvBNReLU(xchannels[0][0], xchannels[0][1], 3, 2, 1)]
        last_channel_idx = 1
        for stage, (t, c, n, s) in enumerate(inverted_residual_setting):
            for i in range(n):
                stride = s if i == 0 else 1
                additv = True if i > 0 else False
                module = block(self.xchannels[last_channel_idx], stride, t, additv)
                features.append(module)
                self.message += '\nstage={:}, ilayer={:02d}/{:02d}, block={:03d}, Cs={:}, stride={:}, expand={:}, original-C={:}'.format(stage, i, n, len(features), self.xchannels[last_channel_idx], stride, t, c)
                last_channel_idx += 1
                if i + 1 == xblocks[stage]:
                    out_channel = module.out_dim
                    for iiL in range(i + 1, n):
                        last_channel_idx += 1
                    self.xchannels[last_channel_idx][0] = module.out_dim
                    break
        features.append(ConvBNReLU(self.xchannels[last_channel_idx][0], self.xchannels[last_channel_idx][1], 1, 1, 1))
        assert last_channel_idx + 2 == len(self.xchannels), '{:} vs {:}'.format(last_channel_idx, len(self.xchannels))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.xchannels[last_channel_idx][1], num_classes))
        self.apply(initialize_resnet)

    def get_message(self):
        return self.message

    def forward(self, inputs):
        features = self.features(inputs)
        vectors = features.mean([2, 3])
        predicts = self.classifier(vectors)
        return features, predicts


class DynamicShapeTinyNet(nn.Module):

    def __init__(self, channels: List[int], genotype: Any, num_classes: int):
        super(DynamicShapeTinyNet, self).__init__()
        self._channels = channels
        if len(channels) % 3 != 2:
            raise ValueError('invalid number of layers : {:}'.format(len(channels)))
        self._num_stage = N = len(channels) // 3
        self.stem = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(channels[0]))
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
        c_prev = channels[0]
        self.cells = nn.ModuleList()
        for index, (c_curr, reduction) in enumerate(zip(channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(c_prev, c_curr, 2, True)
            else:
                cell = InferCell(genotype, c_prev, c_curr, 1)
            self.cells.append(cell)
            c_prev = cell.out_dim
        self._num_layer = len(self.cells)
        self.lastact = nn.Sequential(nn.BatchNorm2d(c_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_prev, num_classes)

    def get_message(self) ->Text:
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return '{name}(C={_channels}, N={_num_stage}, L={_num_layer})'.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)
        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits


def ChannelWiseInterV1(inputs, oC):
    assert inputs.dim() == 4, 'invalid dimension : {:}'.format(inputs.size())

    def start_index(a, b, c):
        return int(math.floor(float(a * c) / b))

    def end_index(a, b, c):
        return int(math.ceil(float((a + 1) * c) / b))
    batch, iC, H, W = inputs.size()
    outputs = torch.zeros((batch, oC, H, W), dtype=inputs.dtype, device=inputs.device)
    if iC == oC:
        return inputs
    for ot in range(oC):
        istartT, iendT = start_index(ot, oC, iC), end_index(ot, oC, iC)
        values = inputs[:, istartT:iendT].mean(dim=1)
        outputs[:, (ot), :, :] = values
    return outputs


def ChannelWiseInterV2(inputs, oC):
    assert inputs.dim() == 4, 'invalid dimension : {:}'.format(inputs.size())
    batch, C, H, W = inputs.size()
    if C == oC:
        return inputs
    else:
        return nn.functional.adaptive_avg_pool3d(inputs, (oC, H, W))


def ChannelWiseInter(inputs, oC, mode='v2'):
    if mode == 'v1':
        return ChannelWiseInterV1(inputs, oC)
    elif mode == 'v2':
        return ChannelWiseInterV2(inputs, oC)
    else:
        raise ValueError('invalid mode : {:}'.format(mode))


def conv_forward(inputs, conv, choices):
    iC = conv.in_channels
    fill_size = list(inputs.size())
    fill_size[1] = iC - fill_size[1]
    filled = torch.zeros(fill_size, device=inputs.device)
    xinputs = torch.cat((inputs, filled), dim=1)
    outputs = conv(xinputs)
    selecteds = [outputs[:, :oC] for oC in choices]
    return selecteds


def get_width_choices(nOut):
    xsrange = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if nOut is None:
        return len(xsrange)
    else:
        Xs = [int(nOut * i) for i in xsrange]
        Xs = sorted(list(set(Xs)))
        return tuple(Xs)


class ConvBNReLU(nn.Module):
    num_conv = 1

    def __init__(self, nIn, nOut, kernel, stride, padding, bias, has_avg, has_bn, has_relu):
        super(ConvBNReLU, self).__init__()
        self.InShape = None
        self.OutShape = None
        self.choices = get_width_choices(nOut)
        self.register_buffer('choices_tensor', torch.Tensor(self.choices))
        if has_avg:
            self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.avg = None
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride, padding=padding, dilation=1, groups=1, bias=bias)
        self.has_bn = has_bn
        self.BNs = nn.ModuleList()
        for i, _out in enumerate(self.choices):
            self.BNs.append(nn.BatchNorm2d(_out))
        if has_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None
        self.in_dim = nIn
        self.out_dim = nOut
        self.search_mode = 'basic'

    def get_flops(self, channels, check_range=True, divide=1):
        iC, oC = channels
        if check_range:
            assert iC <= self.conv.in_channels and oC <= self.conv.out_channels, '{:} vs {:}  |  {:} vs {:}'.format(iC, self.conv.in_channels, oC, self.conv.out_channels)
        assert isinstance(self.InShape, tuple) and len(self.InShape) == 2, 'invalid in-shape : {:}'.format(self.InShape)
        assert isinstance(self.OutShape, tuple) and len(self.OutShape) == 2, 'invalid out-shape : {:}'.format(self.OutShape)
        conv_per_position_flops = self.conv.kernel_size[0] * self.conv.kernel_size[1] * 1.0 / self.conv.groups
        all_positions = self.OutShape[0] * self.OutShape[1]
        flops = conv_per_position_flops * all_positions / divide * iC * oC
        if self.conv.bias is not None:
            flops += all_positions / divide
        return flops

    def get_range(self):
        return [self.choices]

    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

    def search_forward(self, tuple_inputs):
        assert isinstance(tuple_inputs, tuple) and len(tuple_inputs) == 5, 'invalid type input : {:}'.format(type(tuple_inputs))
        inputs, expected_inC, probability, index, prob = tuple_inputs
        index, prob = torch.squeeze(index).tolist(), torch.squeeze(prob)
        probability = torch.squeeze(probability)
        assert len(index) == 2, 'invalid length : {:}'.format(index)
        expected_outC = (self.choices_tensor * probability).sum()
        expected_flop = self.get_flops([expected_inC, expected_outC], False, 1000000.0)
        if self.avg:
            out = self.avg(inputs)
        else:
            out = inputs
        out_convs = conv_forward(out, self.conv, [self.choices[i] for i in index])
        out_bns = [self.BNs[idx](out_conv) for idx, out_conv in zip(index, out_convs)]
        out_channel = max([x.size(1) for x in out_bns])
        outA = ChannelWiseInter(out_bns[0], out_channel)
        outB = ChannelWiseInter(out_bns[1], out_channel)
        out = outA * prob[0] + outB * prob[1]
        if self.relu:
            out = self.relu(out)
        else:
            out = out
        return out, expected_outC, expected_flop

    def basic_forward(self, inputs):
        if self.avg:
            out = self.avg(inputs)
        else:
            out = inputs
        conv = self.conv(out)
        if self.has_bn:
            out = self.BNs[-1](conv)
        else:
            out = conv
        if self.relu:
            out = self.relu(out)
        else:
            out = out
        if self.InShape is None:
            self.InShape = inputs.size(-2), inputs.size(-1)
            self.OutShape = out.size(-2), out.size(-1)
        return out


class ResNetBasicblock(nn.Module):
    expansion = 1
    num_conv = 2

    def __init__(self, inplanes, planes, stride):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ConvBNReLU(inplanes, planes, 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_b = ConvBNReLU(planes, planes, 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=False)
        if stride == 2:
            self.downsample = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=True, has_bn=False, has_relu=False)
        elif inplanes != planes:
            self.downsample = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        else:
            self.downsample = None
        self.out_dim = planes
        self.search_mode = 'basic'

    def get_range(self):
        return self.conv_a.get_range() + self.conv_b.get_range()

    def get_flops(self, channels):
        assert len(channels) == 3, 'invalid channels : {:}'.format(channels)
        flop_A = self.conv_a.get_flops([channels[0], channels[1]])
        flop_B = self.conv_b.get_flops([channels[1], channels[2]])
        if hasattr(self.downsample, 'get_flops'):
            flop_C = self.downsample.get_flops([channels[0], channels[-1]])
        else:
            flop_C = 0
        if channels[0] != channels[-1] and self.downsample is None:
            flop_C = channels[0] * channels[-1] * self.conv_b.OutShape[0] * self.conv_b.OutShape[1]
        return flop_A + flop_B + flop_C

    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

    def search_forward(self, tuple_inputs):
        assert isinstance(tuple_inputs, tuple) and len(tuple_inputs) == 5, 'invalid type input : {:}'.format(type(tuple_inputs))
        inputs, expected_inC, probability, indexes, probs = tuple_inputs
        assert indexes.size(0) == 2 and probs.size(0) == 2 and probability.size(0) == 2
        out_a, expected_inC_a, expected_flop_a = self.conv_a((inputs, expected_inC, probability[0], indexes[0], probs[0]))
        out_b, expected_inC_b, expected_flop_b = self.conv_b((out_a, expected_inC_a, probability[1], indexes[1], probs[1]))
        if self.downsample is not None:
            residual, _, expected_flop_c = self.downsample((inputs, expected_inC, probability[1], indexes[1], probs[1]))
        else:
            residual, expected_flop_c = inputs, 0
        out = additive_func(residual, out_b)
        return nn.functional.relu(out, inplace=True), expected_inC_b, sum([expected_flop_a, expected_flop_b, expected_flop_c])

    def basic_forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = additive_func(residual, basicblock)
        return nn.functional.relu(out, inplace=True)


class ResNetBottleneck(nn.Module):
    expansion = 4
    num_conv = 3

    def __init__(self, inplanes, planes, stride):
        super(ResNetBottleneck, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_1x1 = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_3x3 = ConvBNReLU(planes, planes, 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_1x4 = ConvBNReLU(planes, planes * self.expansion, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        if stride == 2:
            self.downsample = ConvBNReLU(inplanes, planes * self.expansion, 1, 1, 0, False, has_avg=True, has_bn=False, has_relu=False)
        elif inplanes != planes * self.expansion:
            self.downsample = ConvBNReLU(inplanes, planes * self.expansion, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        else:
            self.downsample = None
        self.out_dim = planes * self.expansion
        self.search_mode = 'basic'

    def get_range(self):
        return self.conv_1x1.get_range() + self.conv_3x3.get_range() + self.conv_1x4.get_range()

    def get_flops(self, channels):
        assert len(channels) == 4, 'invalid channels : {:}'.format(channels)
        flop_A = self.conv_1x1.get_flops([channels[0], channels[1]])
        flop_B = self.conv_3x3.get_flops([channels[1], channels[2]])
        flop_C = self.conv_1x4.get_flops([channels[2], channels[3]])
        if hasattr(self.downsample, 'get_flops'):
            flop_D = self.downsample.get_flops([channels[0], channels[-1]])
        else:
            flop_D = 0
        if channels[0] != channels[-1] and self.downsample is None:
            flop_D = channels[0] * channels[-1] * self.conv_1x4.OutShape[0] * self.conv_1x4.OutShape[1]
        return flop_A + flop_B + flop_C + flop_D

    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

    def basic_forward(self, inputs):
        bottleneck = self.conv_1x1(inputs)
        bottleneck = self.conv_3x3(bottleneck)
        bottleneck = self.conv_1x4(bottleneck)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = additive_func(residual, bottleneck)
        return nn.functional.relu(out, inplace=True)

    def search_forward(self, tuple_inputs):
        assert isinstance(tuple_inputs, tuple) and len(tuple_inputs) == 5, 'invalid type input : {:}'.format(type(tuple_inputs))
        inputs, expected_inC, probability, indexes, probs = tuple_inputs
        assert indexes.size(0) == 3 and probs.size(0) == 3 and probability.size(0) == 3
        out_1x1, expected_inC_1x1, expected_flop_1x1 = self.conv_1x1((inputs, expected_inC, probability[0], indexes[0], probs[0]))
        out_3x3, expected_inC_3x3, expected_flop_3x3 = self.conv_3x3((out_1x1, expected_inC_1x1, probability[1], indexes[1], probs[1]))
        out_1x4, expected_inC_1x4, expected_flop_1x4 = self.conv_1x4((out_3x3, expected_inC_3x3, probability[2], indexes[2], probs[2]))
        if self.downsample is not None:
            residual, _, expected_flop_c = self.downsample((inputs, expected_inC, probability[2], indexes[2], probs[2]))
        else:
            residual, expected_flop_c = inputs, 0
        out = additive_func(residual, out_1x4)
        return nn.functional.relu(out, inplace=True), expected_inC_1x4, sum([expected_flop_1x1, expected_flop_3x3, expected_flop_1x4, expected_flop_c])


def get_depth_choices(nDepth):
    if nDepth is None:
        return 3
    else:
        assert nDepth >= 3, 'nDepth should be greater than 2 vs {:}'.format(nDepth)
        if nDepth == 1:
            return 1, 1, 1
        elif nDepth == 2:
            return 1, 1, 2
        elif nDepth >= 3:
            return nDepth // 3, nDepth * 2 // 3, nDepth
        else:
            raise ValueError('invalid Depth : {:}'.format(nDepth))


def linear_forward(inputs, linear):
    if linear is None:
        return inputs
    iC = inputs.size(1)
    weight = linear.weight[:, :iC]
    if linear.bias is None:
        bias = None
    else:
        bias = linear.bias
    return nn.functional.linear(inputs, weight, bias)


def select2withP(logits, tau, just_prob=False, num=2, eps=1e-07):
    if tau <= 0:
        new_logits = logits
        probs = nn.functional.softmax(new_logits, dim=1)
    else:
        while True:
            gumbels = -torch.empty_like(logits).exponential_().log()
            new_logits = (logits.log_softmax(dim=1) + gumbels) / tau
            probs = nn.functional.softmax(new_logits, dim=1)
            if not torch.isinf(gumbels).any() and not torch.isinf(probs).any() and not torch.isnan(probs).any():
                break
    if just_prob:
        return probs
    with torch.no_grad():
        probs = probs.cpu()
        selected_index = torch.multinomial(probs + eps, num, False).to(logits.device)
    selected_logit = torch.gather(new_logits, 1, selected_index)
    selcted_probs = nn.functional.softmax(selected_logit, dim=1)
    return selected_index, selcted_probs


class SearchShapeCifarResNet(nn.Module):

    def __init__(self, block_name, depth, num_classes):
        super(SearchShapeCifarResNet, self).__init__()
        if block_name == 'ResNetBasicblock':
            block = ResNetBasicblock
            assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
            layer_blocks = (depth - 2) // 6
        elif block_name == 'ResNetBottleneck':
            block = ResNetBottleneck
            assert (depth - 2) % 9 == 0, 'depth should be one of 164'
            layer_blocks = (depth - 2) // 9
        else:
            raise ValueError('invalid block : {:}'.format(block_name))
        self.message = 'SearchShapeCifarResNet : Depth : {:} , Layers for each block : {:}'.format(depth, layer_blocks)
        self.num_classes = num_classes
        self.channels = [16]
        self.layers = nn.ModuleList([ConvBNReLU(3, 16, 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=True)])
        self.InShape = None
        self.depth_info = OrderedDict()
        self.depth_at_i = OrderedDict()
        for stage in range(3):
            cur_block_choices = get_depth_choices(layer_blocks, False)
            assert cur_block_choices[-1] == layer_blocks, 'stage={:}, {:} vs {:}'.format(stage, cur_block_choices, layer_blocks)
            self.message += '\nstage={:} ::: depth-block-choices={:} for {:} blocks.'.format(stage, cur_block_choices, layer_blocks)
            block_choices, xstart = [], len(self.layers)
            for iL in range(layer_blocks):
                iC = self.channels[-1]
                planes = 16 * 2 ** stage
                stride = 2 if stage > 0 and iL == 0 else 1
                module = block(iC, planes, stride)
                self.channels.append(module.out_dim)
                self.layers.append(module)
                self.message += '\nstage={:}, ilayer={:02d}/{:02d}, block={:03d}, iC={:3d}, oC={:3d}, stride={:}'.format(stage, iL, layer_blocks, len(self.layers) - 1, iC, module.out_dim, stride)
                layer_index = len(self.layers) - 1
                if iL + 1 in cur_block_choices:
                    block_choices.append(layer_index)
                if iL + 1 == layer_blocks:
                    self.depth_info[layer_index] = {'choices': block_choices, 'stage': stage, 'xstart': xstart}
        self.depth_info_list = []
        for xend, info in self.depth_info.items():
            self.depth_info_list.append((xend, info))
            xstart, xstage = info['xstart'], info['stage']
            for ilayer in range(xstart, xend + 1):
                idx = bisect_right(info['choices'], ilayer - 1)
                self.depth_at_i[ilayer] = xstage, idx
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(module.out_dim, num_classes)
        self.InShape = None
        self.tau = -1
        self.search_mode = 'basic'
        self.Ranges = []
        self.layer2indexRange = []
        for i, layer in enumerate(self.layers):
            start_index = len(self.Ranges)
            self.Ranges += layer.get_range()
            self.layer2indexRange.append((start_index, len(self.Ranges)))
        assert len(self.Ranges) + 1 == depth, 'invalid depth check {:} vs {:}'.format(len(self.Ranges) + 1, depth)
        self.register_parameter('width_attentions', nn.Parameter(torch.Tensor(len(self.Ranges), get_width_choices(None))))
        self.register_parameter('depth_attentions', nn.Parameter(torch.Tensor(3, get_depth_choices(layer_blocks, True))))
        nn.init.normal_(self.width_attentions, 0, 0.01)
        nn.init.normal_(self.depth_attentions, 0, 0.01)
        self.apply(initialize_resnet)

    def arch_parameters(self, LR=None):
        if LR is None:
            return [self.width_attentions, self.depth_attentions]
        else:
            return [{'params': self.width_attentions, 'lr': LR}, {'params': self.depth_attentions, 'lr': LR}]

    def base_parameters(self):
        return list(self.layers.parameters()) + list(self.avgpool.parameters()) + list(self.classifier.parameters())

    def get_flop(self, mode, config_dict, extra_info):
        if config_dict is not None:
            config_dict = config_dict.copy()
        channels = [3]
        for i, weight in enumerate(self.width_attentions):
            if mode == 'genotype':
                with torch.no_grad():
                    probe = nn.functional.softmax(weight, dim=0)
                    C = self.Ranges[i][torch.argmax(probe).item()]
            elif mode == 'max':
                C = self.Ranges[i][-1]
            elif mode == 'fix':
                C = int(math.sqrt(extra_info) * self.Ranges[i][-1])
            elif mode == 'random':
                assert isinstance(extra_info, float), 'invalid extra_info : {:}'.format(extra_info)
                with torch.no_grad():
                    prob = nn.functional.softmax(weight, dim=0)
                    approximate_C = int(math.sqrt(extra_info) * self.Ranges[i][-1])
                    for j in range(prob.size(0)):
                        prob[j] = 1 / (abs(j - (approximate_C - self.Ranges[i][j])) + 0.2)
                    C = self.Ranges[i][torch.multinomial(prob, 1, False).item()]
            else:
                raise ValueError('invalid mode : {:}'.format(mode))
            channels.append(C)
        if mode == 'genotype':
            with torch.no_grad():
                depth_probs = nn.functional.softmax(self.depth_attentions, dim=1)
                choices = torch.argmax(depth_probs, dim=1).cpu().tolist()
        elif mode == 'max' or mode == 'fix':
            choices = [(depth_probs.size(1) - 1) for _ in range(depth_probs.size(0))]
        elif mode == 'random':
            with torch.no_grad():
                depth_probs = nn.functional.softmax(self.depth_attentions, dim=1)
                choices = torch.multinomial(depth_probs, 1, False).cpu().tolist()
        else:
            raise ValueError('invalid mode : {:}'.format(mode))
        selected_layers = []
        for choice, xvalue in zip(choices, self.depth_info_list):
            xtemp = xvalue[1]['choices'][choice] - xvalue[1]['xstart'] + 1
            selected_layers.append(xtemp)
        flop = 0
        for i, layer in enumerate(self.layers):
            s, e = self.layer2indexRange[i]
            xchl = tuple(channels[s:e + 1])
            if i in self.depth_at_i:
                xstagei, xatti = self.depth_at_i[i]
                if xatti <= choices[xstagei]:
                    flop += layer.get_flops(xchl)
                else:
                    flop += 0
            else:
                flop += layer.get_flops(xchl)
        flop += channels[-1] * self.classifier.out_features
        if config_dict is None:
            return flop / 1000000.0
        else:
            config_dict['xchannels'] = channels
            config_dict['xblocks'] = selected_layers
            config_dict['super_type'] = 'infer-shape'
            config_dict['estimated_FLOP'] = flop / 1000000.0
            return flop / 1000000.0, config_dict

    def get_arch_info(self):
        string = 'for depth and width, there are {:} + {:} attention probabilities.'.format(len(self.depth_attentions), len(self.width_attentions))
        string += '\n{:}'.format(self.depth_info)
        discrepancy = []
        with torch.no_grad():
            for i, att in enumerate(self.depth_attentions):
                prob = nn.functional.softmax(att, dim=0)
                prob = prob.cpu()
                selc = prob.argmax().item()
                prob = prob.tolist()
                prob = ['{:.3f}'.format(x) for x in prob]
                xstring = '{:03d}/{:03d}-th : {:}'.format(i, len(self.depth_attentions), ' '.join(prob))
                logt = ['{:.4f}'.format(x) for x in att.cpu().tolist()]
                xstring += '  ||  {:17s}'.format(' '.join(logt))
                prob = sorted([float(x) for x in prob])
                disc = prob[-1] - prob[-2]
                xstring += '  || discrepancy={:.2f} || select={:}/{:}'.format(disc, selc, len(prob))
                discrepancy.append(disc)
                string += '\n{:}'.format(xstring)
            string += '\n-----------------------------------------------'
            for i, att in enumerate(self.width_attentions):
                prob = nn.functional.softmax(att, dim=0)
                prob = prob.cpu()
                selc = prob.argmax().item()
                prob = prob.tolist()
                prob = ['{:.3f}'.format(x) for x in prob]
                xstring = '{:03d}/{:03d}-th : {:}'.format(i, len(self.width_attentions), ' '.join(prob))
                logt = ['{:.3f}'.format(x) for x in att.cpu().tolist()]
                xstring += '  ||  {:52s}'.format(' '.join(logt))
                prob = sorted([float(x) for x in prob])
                disc = prob[-1] - prob[-2]
                xstring += '  || dis={:.2f} || select={:}/{:}'.format(disc, selc, len(prob))
                discrepancy.append(disc)
                string += '\n{:}'.format(xstring)
        return string, discrepancy

    def set_tau(self, tau_max, tau_min, epoch_ratio):
        assert epoch_ratio >= 0 and epoch_ratio <= 1, 'invalid epoch-ratio : {:}'.format(epoch_ratio)
        tau = tau_min + (tau_max - tau_min) * (1 + math.cos(math.pi * epoch_ratio)) / 2
        self.tau = tau

    def get_message(self):
        return self.message

    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

    def search_forward(self, inputs):
        flop_width_probs = nn.functional.softmax(self.width_attentions, dim=1)
        flop_depth_probs = nn.functional.softmax(self.depth_attentions, dim=1)
        flop_depth_probs = torch.flip(torch.cumsum(torch.flip(flop_depth_probs, [1]), 1), [1])
        selected_widths, selected_width_probs = select2withP(self.width_attentions, self.tau)
        selected_depth_probs = select2withP(self.depth_attentions, self.tau, True)
        with torch.no_grad():
            selected_widths = selected_widths.cpu()
        x, last_channel_idx, expected_inC, flops = inputs, 0, 3, []
        feature_maps = []
        for i, layer in enumerate(self.layers):
            selected_w_index = selected_widths[last_channel_idx:last_channel_idx + layer.num_conv]
            selected_w_probs = selected_width_probs[last_channel_idx:last_channel_idx + layer.num_conv]
            layer_prob = flop_width_probs[last_channel_idx:last_channel_idx + layer.num_conv]
            x, expected_inC, expected_flop = layer((x, expected_inC, layer_prob, selected_w_index, selected_w_probs))
            feature_maps.append(x)
            last_channel_idx += layer.num_conv
            if i in self.depth_info:
                choices = self.depth_info[i]['choices']
                xstagei = self.depth_info[i]['stage']
                possible_tensors = []
                max_C = max(feature_maps[A].size(1) for A in choices)
                for tempi, A in enumerate(choices):
                    xtensor = ChannelWiseInter(feature_maps[A], max_C)
                    possible_tensors.append(xtensor)
                weighted_sum = sum(xtensor * W for xtensor, W in zip(possible_tensors, selected_depth_probs[xstagei]))
                x = weighted_sum
            if i in self.depth_at_i:
                xstagei, xatti = self.depth_at_i[i]
                x_expected_flop = flop_depth_probs[xstagei, xatti] * expected_flop
            else:
                x_expected_flop = expected_flop
            flops.append(x_expected_flop)
        flops.append(expected_inC * (self.classifier.out_features * 1.0 / 1000000.0))
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = linear_forward(features, self.classifier)
        return logits, torch.stack([sum(flops)])

    def basic_forward(self, inputs):
        if self.InShape is None:
            self.InShape = inputs.size(-2), inputs.size(-1)
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return features, logits


class ConvBNReLU(nn.Module):
    num_conv = 1

    def __init__(self, nIn, nOut, kernel, stride, padding, bias, has_avg, has_bn, has_relu):
        super(ConvBNReLU, self).__init__()
        self.InShape = None
        self.OutShape = None
        self.choices = get_width_choices(nOut)
        self.register_buffer('choices_tensor', torch.Tensor(self.choices))
        if has_avg:
            self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.avg = None
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride, padding=padding, dilation=1, groups=1, bias=bias)
        if has_bn:
            self.bn = nn.BatchNorm2d(nOut)
        else:
            self.bn = None
        if has_relu:
            self.relu = nn.ReLU(inplace=False)
        else:
            self.relu = None
        self.in_dim = nIn
        self.out_dim = nOut

    def get_flops(self, divide=1):
        iC, oC = self.in_dim, self.out_dim
        assert iC <= self.conv.in_channels and oC <= self.conv.out_channels, '{:} vs {:}  |  {:} vs {:}'.format(iC, self.conv.in_channels, oC, self.conv.out_channels)
        assert isinstance(self.InShape, tuple) and len(self.InShape) == 2, 'invalid in-shape : {:}'.format(self.InShape)
        assert isinstance(self.OutShape, tuple) and len(self.OutShape) == 2, 'invalid out-shape : {:}'.format(self.OutShape)
        conv_per_position_flops = self.conv.kernel_size[0] * self.conv.kernel_size[1] * 1.0 / self.conv.groups
        all_positions = self.OutShape[0] * self.OutShape[1]
        flops = conv_per_position_flops * all_positions / divide * iC * oC
        if self.conv.bias is not None:
            flops += all_positions / divide
        return flops

    def forward(self, inputs):
        if self.avg:
            out = self.avg(inputs)
        else:
            out = inputs
        conv = self.conv(out)
        if self.bn:
            out = self.bn(conv)
        else:
            out = conv
        if self.relu:
            out = self.relu(out)
        else:
            out = out
        if self.InShape is None:
            self.InShape = inputs.size(-2), inputs.size(-1)
            self.OutShape = out.size(-2), out.size(-1)
        return out


class ResNetBasicblock(nn.Module):
    expansion = 1
    num_conv = 2

    def __init__(self, inplanes, planes, stride):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ConvBNReLU(inplanes, planes, 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_b = ConvBNReLU(planes, planes, 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=False)
        if stride == 2:
            self.downsample = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=True, has_bn=False, has_relu=False)
        elif inplanes != planes:
            self.downsample = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        else:
            self.downsample = None
        self.out_dim = planes
        self.search_mode = 'basic'

    def get_flops(self, divide=1):
        flop_A = self.conv_a.get_flops(divide)
        flop_B = self.conv_b.get_flops(divide)
        if hasattr(self.downsample, 'get_flops'):
            flop_C = self.downsample.get_flops(divide)
        else:
            flop_C = 0
        return flop_A + flop_B + flop_C

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = additive_func(residual, basicblock)
        return nn.functional.relu(out, inplace=True)


class ResNetBottleneck(nn.Module):
    expansion = 4
    num_conv = 3

    def __init__(self, inplanes, planes, stride):
        super(ResNetBottleneck, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_1x1 = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_3x3 = ConvBNReLU(planes, planes, 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_1x4 = ConvBNReLU(planes, planes * self.expansion, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        if stride == 2:
            self.downsample = ConvBNReLU(inplanes, planes * self.expansion, 1, 1, 0, False, has_avg=True, has_bn=False, has_relu=False)
        elif inplanes != planes * self.expansion:
            self.downsample = ConvBNReLU(inplanes, planes * self.expansion, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        else:
            self.downsample = None
        self.out_dim = planes * self.expansion
        self.search_mode = 'basic'

    def get_range(self):
        return self.conv_1x1.get_range() + self.conv_3x3.get_range() + self.conv_1x4.get_range()

    def get_flops(self, divide):
        flop_A = self.conv_1x1.get_flops(divide)
        flop_B = self.conv_3x3.get_flops(divide)
        flop_C = self.conv_1x4.get_flops(divide)
        if hasattr(self.downsample, 'get_flops'):
            flop_D = self.downsample.get_flops(divide)
        else:
            flop_D = 0
        return flop_A + flop_B + flop_C + flop_D

    def forward(self, inputs):
        bottleneck = self.conv_1x1(inputs)
        bottleneck = self.conv_3x3(bottleneck)
        bottleneck = self.conv_1x4(bottleneck)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = additive_func(residual, bottleneck)
        return nn.functional.relu(out, inplace=True)


class SearchDepthCifarResNet(nn.Module):

    def __init__(self, block_name, depth, num_classes):
        super(SearchDepthCifarResNet, self).__init__()
        if block_name == 'ResNetBasicblock':
            block = ResNetBasicblock
            assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
            layer_blocks = (depth - 2) // 6
        elif block_name == 'ResNetBottleneck':
            block = ResNetBottleneck
            assert (depth - 2) % 9 == 0, 'depth should be one of 164'
            layer_blocks = (depth - 2) // 9
        else:
            raise ValueError('invalid block : {:}'.format(block_name))
        self.message = 'SearchShapeCifarResNet : Depth : {:} , Layers for each block : {:}'.format(depth, layer_blocks)
        self.num_classes = num_classes
        self.channels = [16]
        self.layers = nn.ModuleList([ConvBNReLU(3, 16, 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=True)])
        self.InShape = None
        self.depth_info = OrderedDict()
        self.depth_at_i = OrderedDict()
        for stage in range(3):
            cur_block_choices = get_depth_choices(layer_blocks, False)
            assert cur_block_choices[-1] == layer_blocks, 'stage={:}, {:} vs {:}'.format(stage, cur_block_choices, layer_blocks)
            self.message += '\nstage={:} ::: depth-block-choices={:} for {:} blocks.'.format(stage, cur_block_choices, layer_blocks)
            block_choices, xstart = [], len(self.layers)
            for iL in range(layer_blocks):
                iC = self.channels[-1]
                planes = 16 * 2 ** stage
                stride = 2 if stage > 0 and iL == 0 else 1
                module = block(iC, planes, stride)
                self.channels.append(module.out_dim)
                self.layers.append(module)
                self.message += '\nstage={:}, ilayer={:02d}/{:02d}, block={:03d}, iC={:3d}, oC={:3d}, stride={:}'.format(stage, iL, layer_blocks, len(self.layers) - 1, iC, module.out_dim, stride)
                layer_index = len(self.layers) - 1
                if iL + 1 in cur_block_choices:
                    block_choices.append(layer_index)
                if iL + 1 == layer_blocks:
                    self.depth_info[layer_index] = {'choices': block_choices, 'stage': stage, 'xstart': xstart}
        self.depth_info_list = []
        for xend, info in self.depth_info.items():
            self.depth_info_list.append((xend, info))
            xstart, xstage = info['xstart'], info['stage']
            for ilayer in range(xstart, xend + 1):
                idx = bisect_right(info['choices'], ilayer - 1)
                self.depth_at_i[ilayer] = xstage, idx
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(module.out_dim, num_classes)
        self.InShape = None
        self.tau = -1
        self.search_mode = 'basic'
        self.register_parameter('depth_attentions', nn.Parameter(torch.Tensor(3, get_depth_choices(layer_blocks, True))))
        nn.init.normal_(self.depth_attentions, 0, 0.01)
        self.apply(initialize_resnet)

    def arch_parameters(self):
        return [self.depth_attentions]

    def base_parameters(self):
        return list(self.layers.parameters()) + list(self.avgpool.parameters()) + list(self.classifier.parameters())

    def get_flop(self, mode, config_dict, extra_info):
        if config_dict is not None:
            config_dict = config_dict.copy()
        if mode == 'genotype':
            with torch.no_grad():
                depth_probs = nn.functional.softmax(self.depth_attentions, dim=1)
                choices = torch.argmax(depth_probs, dim=1).cpu().tolist()
        elif mode == 'max':
            choices = [(depth_probs.size(1) - 1) for _ in range(depth_probs.size(0))]
        elif mode == 'random':
            with torch.no_grad():
                depth_probs = nn.functional.softmax(self.depth_attentions, dim=1)
                choices = torch.multinomial(depth_probs, 1, False).cpu().tolist()
        else:
            raise ValueError('invalid mode : {:}'.format(mode))
        selected_layers = []
        for choice, xvalue in zip(choices, self.depth_info_list):
            xtemp = xvalue[1]['choices'][choice] - xvalue[1]['xstart'] + 1
            selected_layers.append(xtemp)
        flop = 0
        for i, layer in enumerate(self.layers):
            if i in self.depth_at_i:
                xstagei, xatti = self.depth_at_i[i]
                if xatti <= choices[xstagei]:
                    flop += layer.get_flops()
                else:
                    flop += 0
            else:
                flop += layer.get_flops()
        flop += self.classifier.in_features * self.classifier.out_features
        if config_dict is None:
            return flop / 1000000.0
        else:
            config_dict['xblocks'] = selected_layers
            config_dict['super_type'] = 'infer-depth'
            config_dict['estimated_FLOP'] = flop / 1000000.0
            return flop / 1000000.0, config_dict

    def get_arch_info(self):
        string = 'for depth, there are {:} attention probabilities.'.format(len(self.depth_attentions))
        string += '\n{:}'.format(self.depth_info)
        discrepancy = []
        with torch.no_grad():
            for i, att in enumerate(self.depth_attentions):
                prob = nn.functional.softmax(att, dim=0)
                prob = prob.cpu()
                selc = prob.argmax().item()
                prob = prob.tolist()
                prob = ['{:.3f}'.format(x) for x in prob]
                xstring = '{:03d}/{:03d}-th : {:}'.format(i, len(self.depth_attentions), ' '.join(prob))
                logt = ['{:.4f}'.format(x) for x in att.cpu().tolist()]
                xstring += '  ||  {:17s}'.format(' '.join(logt))
                prob = sorted([float(x) for x in prob])
                disc = prob[-1] - prob[-2]
                xstring += '  || discrepancy={:.2f} || select={:}/{:}'.format(disc, selc, len(prob))
                discrepancy.append(disc)
                string += '\n{:}'.format(xstring)
        return string, discrepancy

    def set_tau(self, tau_max, tau_min, epoch_ratio):
        assert epoch_ratio >= 0 and epoch_ratio <= 1, 'invalid epoch-ratio : {:}'.format(epoch_ratio)
        tau = tau_min + (tau_max - tau_min) * (1 + math.cos(math.pi * epoch_ratio)) / 2
        self.tau = tau

    def get_message(self):
        return self.message

    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

    def search_forward(self, inputs):
        flop_depth_probs = nn.functional.softmax(self.depth_attentions, dim=1)
        flop_depth_probs = torch.flip(torch.cumsum(torch.flip(flop_depth_probs, [1]), 1), [1])
        selected_depth_probs = select2withP(self.depth_attentions, self.tau, True)
        x, flops = inputs, []
        feature_maps = []
        for i, layer in enumerate(self.layers):
            layer_i = layer(x)
            feature_maps.append(layer_i)
            if i in self.depth_info:
                choices = self.depth_info[i]['choices']
                xstagei = self.depth_info[i]['stage']
                possible_tensors = []
                for tempi, A in enumerate(choices):
                    xtensor = feature_maps[A]
                    possible_tensors.append(xtensor)
                weighted_sum = sum(xtensor * W for xtensor, W in zip(possible_tensors, selected_depth_probs[xstagei]))
                x = weighted_sum
            else:
                x = layer_i
            if i in self.depth_at_i:
                xstagei, xatti = self.depth_at_i[i]
                x_expected_flop = flop_depth_probs[xstagei, xatti] * layer.get_flops(1000000.0)
            else:
                x_expected_flop = layer.get_flops(1000000.0)
            flops.append(x_expected_flop)
        flops.append(self.classifier.in_features * self.classifier.out_features * 1.0 / 1000000.0)
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = linear_forward(features, self.classifier)
        return logits, torch.stack([sum(flops)])

    def basic_forward(self, inputs):
        if self.InShape is None:
            self.InShape = inputs.size(-2), inputs.size(-1)
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return features, logits


class ConvBNReLU(nn.Module):
    num_conv = 1

    def __init__(self, nIn, nOut, kernel, stride, padding, bias, has_avg, has_bn, has_relu):
        super(ConvBNReLU, self).__init__()
        self.InShape = None
        self.OutShape = None
        self.choices = get_choices(nOut)
        self.register_buffer('choices_tensor', torch.Tensor(self.choices))
        if has_avg:
            self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.avg = None
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride, padding=padding, dilation=1, groups=1, bias=bias)
        self.has_bn = has_bn
        self.BNs = nn.ModuleList()
        for i, _out in enumerate(self.choices):
            self.BNs.append(nn.BatchNorm2d(_out))
        if has_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None
        self.in_dim = nIn
        self.out_dim = nOut
        self.search_mode = 'basic'

    def get_flops(self, channels, check_range=True, divide=1):
        iC, oC = channels
        if check_range:
            assert iC <= self.conv.in_channels and oC <= self.conv.out_channels, '{:} vs {:}  |  {:} vs {:}'.format(iC, self.conv.in_channels, oC, self.conv.out_channels)
        assert isinstance(self.InShape, tuple) and len(self.InShape) == 2, 'invalid in-shape : {:}'.format(self.InShape)
        assert isinstance(self.OutShape, tuple) and len(self.OutShape) == 2, 'invalid out-shape : {:}'.format(self.OutShape)
        conv_per_position_flops = self.conv.kernel_size[0] * self.conv.kernel_size[1] * 1.0 / self.conv.groups
        all_positions = self.OutShape[0] * self.OutShape[1]
        flops = conv_per_position_flops * all_positions / divide * iC * oC
        if self.conv.bias is not None:
            flops += all_positions / divide
        return flops

    def get_range(self):
        return [self.choices]

    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

    def search_forward(self, tuple_inputs):
        assert isinstance(tuple_inputs, tuple) and len(tuple_inputs) == 5, 'invalid type input : {:}'.format(type(tuple_inputs))
        inputs, expected_inC, probability, index, prob = tuple_inputs
        index, prob = torch.squeeze(index).tolist(), torch.squeeze(prob)
        probability = torch.squeeze(probability)
        assert len(index) == 2, 'invalid length : {:}'.format(index)
        expected_outC = (self.choices_tensor * probability).sum()
        expected_flop = self.get_flops([expected_inC, expected_outC], False, 1000000.0)
        if self.avg:
            out = self.avg(inputs)
        else:
            out = inputs
        out_convs = conv_forward(out, self.conv, [self.choices[i] for i in index])
        out_bns = [self.BNs[idx](out_conv) for idx, out_conv in zip(index, out_convs)]
        out_channel = max([x.size(1) for x in out_bns])
        outA = ChannelWiseInter(out_bns[0], out_channel)
        outB = ChannelWiseInter(out_bns[1], out_channel)
        out = outA * prob[0] + outB * prob[1]
        if self.relu:
            out = self.relu(out)
        else:
            out = out
        return out, expected_outC, expected_flop

    def basic_forward(self, inputs):
        if self.avg:
            out = self.avg(inputs)
        else:
            out = inputs
        conv = self.conv(out)
        if self.has_bn:
            out = self.BNs[-1](conv)
        else:
            out = conv
        if self.relu:
            out = self.relu(out)
        else:
            out = out
        if self.InShape is None:
            self.InShape = inputs.size(-2), inputs.size(-1)
            self.OutShape = out.size(-2), out.size(-1)
        return out


class ResNetBasicblock(nn.Module):
    expansion = 1
    num_conv = 2

    def __init__(self, inplanes, planes, stride):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ConvBNReLU(inplanes, planes, 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_b = ConvBNReLU(planes, planes, 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=False)
        if stride == 2:
            self.downsample = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=True, has_bn=False, has_relu=False)
        elif inplanes != planes:
            self.downsample = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        else:
            self.downsample = None
        self.out_dim = planes
        self.search_mode = 'basic'

    def get_range(self):
        return self.conv_a.get_range() + self.conv_b.get_range()

    def get_flops(self, channels):
        assert len(channels) == 3, 'invalid channels : {:}'.format(channels)
        flop_A = self.conv_a.get_flops([channels[0], channels[1]])
        flop_B = self.conv_b.get_flops([channels[1], channels[2]])
        if hasattr(self.downsample, 'get_flops'):
            flop_C = self.downsample.get_flops([channels[0], channels[-1]])
        else:
            flop_C = 0
        if channels[0] != channels[-1] and self.downsample is None:
            flop_C = channels[0] * channels[-1] * self.conv_b.OutShape[0] * self.conv_b.OutShape[1]
        return flop_A + flop_B + flop_C

    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

    def search_forward(self, tuple_inputs):
        assert isinstance(tuple_inputs, tuple) and len(tuple_inputs) == 5, 'invalid type input : {:}'.format(type(tuple_inputs))
        inputs, expected_inC, probability, indexes, probs = tuple_inputs
        assert indexes.size(0) == 2 and probs.size(0) == 2 and probability.size(0) == 2
        out_a, expected_inC_a, expected_flop_a = self.conv_a((inputs, expected_inC, probability[0], indexes[0], probs[0]))
        out_b, expected_inC_b, expected_flop_b = self.conv_b((out_a, expected_inC_a, probability[1], indexes[1], probs[1]))
        if self.downsample is not None:
            residual, _, expected_flop_c = self.downsample((inputs, expected_inC, probability[1], indexes[1], probs[1]))
        else:
            residual, expected_flop_c = inputs, 0
        out = additive_func(residual, out_b)
        return nn.functional.relu(out, inplace=True), expected_inC_b, sum([expected_flop_a, expected_flop_b, expected_flop_c])

    def basic_forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = additive_func(residual, basicblock)
        return nn.functional.relu(out, inplace=True)


class ResNetBottleneck(nn.Module):
    expansion = 4
    num_conv = 3

    def __init__(self, inplanes, planes, stride):
        super(ResNetBottleneck, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_1x1 = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_3x3 = ConvBNReLU(planes, planes, 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_1x4 = ConvBNReLU(planes, planes * self.expansion, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        if stride == 2:
            self.downsample = ConvBNReLU(inplanes, planes * self.expansion, 1, 1, 0, False, has_avg=True, has_bn=False, has_relu=False)
        elif inplanes != planes * self.expansion:
            self.downsample = ConvBNReLU(inplanes, planes * self.expansion, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        else:
            self.downsample = None
        self.out_dim = planes * self.expansion
        self.search_mode = 'basic'

    def get_range(self):
        return self.conv_1x1.get_range() + self.conv_3x3.get_range() + self.conv_1x4.get_range()

    def get_flops(self, channels):
        assert len(channels) == 4, 'invalid channels : {:}'.format(channels)
        flop_A = self.conv_1x1.get_flops([channels[0], channels[1]])
        flop_B = self.conv_3x3.get_flops([channels[1], channels[2]])
        flop_C = self.conv_1x4.get_flops([channels[2], channels[3]])
        if hasattr(self.downsample, 'get_flops'):
            flop_D = self.downsample.get_flops([channels[0], channels[-1]])
        else:
            flop_D = 0
        if channels[0] != channels[-1] and self.downsample is None:
            flop_D = channels[0] * channels[-1] * self.conv_1x4.OutShape[0] * self.conv_1x4.OutShape[1]
        return flop_A + flop_B + flop_C + flop_D

    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

    def basic_forward(self, inputs):
        bottleneck = self.conv_1x1(inputs)
        bottleneck = self.conv_3x3(bottleneck)
        bottleneck = self.conv_1x4(bottleneck)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = additive_func(residual, bottleneck)
        return nn.functional.relu(out, inplace=True)

    def search_forward(self, tuple_inputs):
        assert isinstance(tuple_inputs, tuple) and len(tuple_inputs) == 5, 'invalid type input : {:}'.format(type(tuple_inputs))
        inputs, expected_inC, probability, indexes, probs = tuple_inputs
        assert indexes.size(0) == 3 and probs.size(0) == 3 and probability.size(0) == 3
        out_1x1, expected_inC_1x1, expected_flop_1x1 = self.conv_1x1((inputs, expected_inC, probability[0], indexes[0], probs[0]))
        out_3x3, expected_inC_3x3, expected_flop_3x3 = self.conv_3x3((out_1x1, expected_inC_1x1, probability[1], indexes[1], probs[1]))
        out_1x4, expected_inC_1x4, expected_flop_1x4 = self.conv_1x4((out_3x3, expected_inC_3x3, probability[2], indexes[2], probs[2]))
        if self.downsample is not None:
            residual, _, expected_flop_c = self.downsample((inputs, expected_inC, probability[2], indexes[2], probs[2]))
        else:
            residual, expected_flop_c = inputs, 0
        out = additive_func(residual, out_1x4)
        return nn.functional.relu(out, inplace=True), expected_inC_1x4, sum([expected_flop_1x1, expected_flop_3x3, expected_flop_1x4, expected_flop_c])


class SearchWidthCifarResNet(nn.Module):

    def __init__(self, block_name, depth, num_classes):
        super(SearchWidthCifarResNet, self).__init__()
        if block_name == 'ResNetBasicblock':
            block = ResNetBasicblock
            assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
            layer_blocks = (depth - 2) // 6
        elif block_name == 'ResNetBottleneck':
            block = ResNetBottleneck
            assert (depth - 2) % 9 == 0, 'depth should be one of 164'
            layer_blocks = (depth - 2) // 9
        else:
            raise ValueError('invalid block : {:}'.format(block_name))
        self.message = 'SearchWidthCifarResNet : Depth : {:} , Layers for each block : {:}'.format(depth, layer_blocks)
        self.num_classes = num_classes
        self.channels = [16]
        self.layers = nn.ModuleList([ConvBNReLU(3, 16, 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=True)])
        self.InShape = None
        for stage in range(3):
            for iL in range(layer_blocks):
                iC = self.channels[-1]
                planes = 16 * 2 ** stage
                stride = 2 if stage > 0 and iL == 0 else 1
                module = block(iC, planes, stride)
                self.channels.append(module.out_dim)
                self.layers.append(module)
                self.message += '\nstage={:}, ilayer={:02d}/{:02d}, block={:03d}, iC={:3d}, oC={:3d}, stride={:}'.format(stage, iL, layer_blocks, len(self.layers) - 1, iC, module.out_dim, stride)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(module.out_dim, num_classes)
        self.InShape = None
        self.tau = -1
        self.search_mode = 'basic'
        self.Ranges = []
        self.layer2indexRange = []
        for i, layer in enumerate(self.layers):
            start_index = len(self.Ranges)
            self.Ranges += layer.get_range()
            self.layer2indexRange.append((start_index, len(self.Ranges)))
        assert len(self.Ranges) + 1 == depth, 'invalid depth check {:} vs {:}'.format(len(self.Ranges) + 1, depth)
        self.register_parameter('width_attentions', nn.Parameter(torch.Tensor(len(self.Ranges), get_choices(None))))
        nn.init.normal_(self.width_attentions, 0, 0.01)
        self.apply(initialize_resnet)

    def arch_parameters(self):
        return [self.width_attentions]

    def base_parameters(self):
        return list(self.layers.parameters()) + list(self.avgpool.parameters()) + list(self.classifier.parameters())

    def get_flop(self, mode, config_dict, extra_info):
        if config_dict is not None:
            config_dict = config_dict.copy()
        channels = [3]
        for i, weight in enumerate(self.width_attentions):
            if mode == 'genotype':
                with torch.no_grad():
                    probe = nn.functional.softmax(weight, dim=0)
                    C = self.Ranges[i][torch.argmax(probe).item()]
            elif mode == 'max':
                C = self.Ranges[i][-1]
            elif mode == 'fix':
                C = int(math.sqrt(extra_info) * self.Ranges[i][-1])
            elif mode == 'random':
                assert isinstance(extra_info, float), 'invalid extra_info : {:}'.format(extra_info)
                with torch.no_grad():
                    prob = nn.functional.softmax(weight, dim=0)
                    approximate_C = int(math.sqrt(extra_info) * self.Ranges[i][-1])
                    for j in range(prob.size(0)):
                        prob[j] = 1 / (abs(j - (approximate_C - self.Ranges[i][j])) + 0.2)
                    C = self.Ranges[i][torch.multinomial(prob, 1, False).item()]
            else:
                raise ValueError('invalid mode : {:}'.format(mode))
            channels.append(C)
        flop = 0
        for i, layer in enumerate(self.layers):
            s, e = self.layer2indexRange[i]
            xchl = tuple(channels[s:e + 1])
            flop += layer.get_flops(xchl)
        flop += channels[-1] * self.classifier.out_features
        if config_dict is None:
            return flop / 1000000.0
        else:
            config_dict['xchannels'] = channels
            config_dict['super_type'] = 'infer-width'
            config_dict['estimated_FLOP'] = flop / 1000000.0
            return flop / 1000000.0, config_dict

    def get_arch_info(self):
        string = 'for width, there are {:} attention probabilities.'.format(len(self.width_attentions))
        discrepancy = []
        with torch.no_grad():
            for i, att in enumerate(self.width_attentions):
                prob = nn.functional.softmax(att, dim=0)
                prob = prob.cpu()
                selc = prob.argmax().item()
                prob = prob.tolist()
                prob = ['{:.3f}'.format(x) for x in prob]
                xstring = '{:03d}/{:03d}-th : {:}'.format(i, len(self.width_attentions), ' '.join(prob))
                logt = ['{:.3f}'.format(x) for x in att.cpu().tolist()]
                xstring += '  ||  {:52s}'.format(' '.join(logt))
                prob = sorted([float(x) for x in prob])
                disc = prob[-1] - prob[-2]
                xstring += '  || dis={:.2f} || select={:}/{:}'.format(disc, selc, len(prob))
                discrepancy.append(disc)
                string += '\n{:}'.format(xstring)
        return string, discrepancy

    def set_tau(self, tau_max, tau_min, epoch_ratio):
        assert epoch_ratio >= 0 and epoch_ratio <= 1, 'invalid epoch-ratio : {:}'.format(epoch_ratio)
        tau = tau_min + (tau_max - tau_min) * (1 + math.cos(math.pi * epoch_ratio)) / 2
        self.tau = tau

    def get_message(self):
        return self.message

    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

    def search_forward(self, inputs):
        flop_probs = nn.functional.softmax(self.width_attentions, dim=1)
        selected_widths, selected_probs = select2withP(self.width_attentions, self.tau)
        with torch.no_grad():
            selected_widths = selected_widths.cpu()
        x, last_channel_idx, expected_inC, flops = inputs, 0, 3, []
        for i, layer in enumerate(self.layers):
            selected_w_index = selected_widths[last_channel_idx:last_channel_idx + layer.num_conv]
            selected_w_probs = selected_probs[last_channel_idx:last_channel_idx + layer.num_conv]
            layer_prob = flop_probs[last_channel_idx:last_channel_idx + layer.num_conv]
            x, expected_inC, expected_flop = layer((x, expected_inC, layer_prob, selected_w_index, selected_w_probs))
            last_channel_idx += layer.num_conv
            flops.append(expected_flop)
        flops.append(expected_inC * (self.classifier.out_features * 1.0 / 1000000.0))
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = linear_forward(features, self.classifier)
        return logits, torch.stack([sum(flops)])

    def basic_forward(self, inputs):
        if self.InShape is None:
            self.InShape = inputs.size(-2), inputs.size(-1)
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return features, logits


class ConvBNReLU(nn.Module):
    num_conv = 1

    def __init__(self, nIn, nOut, kernel, stride, padding, bias, has_avg, has_bn, has_relu, last_max_pool=False):
        super(ConvBNReLU, self).__init__()
        self.InShape = None
        self.OutShape = None
        self.choices = get_width_choices(nOut)
        self.register_buffer('choices_tensor', torch.Tensor(self.choices))
        if has_avg:
            self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.avg = None
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride, padding=padding, dilation=1, groups=1, bias=bias)
        self.has_bn = has_bn
        self.BNs = nn.ModuleList()
        for i, _out in enumerate(self.choices):
            self.BNs.append(nn.BatchNorm2d(_out))
        if has_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None
        if last_max_pool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = None
        self.in_dim = nIn
        self.out_dim = nOut
        self.search_mode = 'basic'

    def get_flops(self, channels, check_range=True, divide=1):
        iC, oC = channels
        if check_range:
            assert iC <= self.conv.in_channels and oC <= self.conv.out_channels, '{:} vs {:}  |  {:} vs {:}'.format(iC, self.conv.in_channels, oC, self.conv.out_channels)
        assert isinstance(self.InShape, tuple) and len(self.InShape) == 2, 'invalid in-shape : {:}'.format(self.InShape)
        assert isinstance(self.OutShape, tuple) and len(self.OutShape) == 2, 'invalid out-shape : {:}'.format(self.OutShape)
        conv_per_position_flops = self.conv.kernel_size[0] * self.conv.kernel_size[1] * 1.0 / self.conv.groups
        all_positions = self.OutShape[0] * self.OutShape[1]
        flops = conv_per_position_flops * all_positions / divide * iC * oC
        if self.conv.bias is not None:
            flops += all_positions / divide
        return flops

    def get_range(self):
        return [self.choices]

    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

    def search_forward(self, tuple_inputs):
        assert isinstance(tuple_inputs, tuple) and len(tuple_inputs) == 5, 'invalid type input : {:}'.format(type(tuple_inputs))
        inputs, expected_inC, probability, index, prob = tuple_inputs
        index, prob = torch.squeeze(index).tolist(), torch.squeeze(prob)
        probability = torch.squeeze(probability)
        assert len(index) == 2, 'invalid length : {:}'.format(index)
        expected_outC = (self.choices_tensor * probability).sum()
        expected_flop = self.get_flops([expected_inC, expected_outC], False, 1000000.0)
        if self.avg:
            out = self.avg(inputs)
        else:
            out = inputs
        out_convs = conv_forward(out, self.conv, [self.choices[i] for i in index])
        out_bns = [self.BNs[idx](out_conv) for idx, out_conv in zip(index, out_convs)]
        out_channel = max([x.size(1) for x in out_bns])
        outA = ChannelWiseInter(out_bns[0], out_channel)
        outB = ChannelWiseInter(out_bns[1], out_channel)
        out = outA * prob[0] + outB * prob[1]
        if self.relu:
            out = self.relu(out)
        if self.maxpool:
            out = self.maxpool(out)
        return out, expected_outC, expected_flop

    def basic_forward(self, inputs):
        if self.avg:
            out = self.avg(inputs)
        else:
            out = inputs
        conv = self.conv(out)
        if self.has_bn:
            out = self.BNs[-1](conv)
        else:
            out = conv
        if self.relu:
            out = self.relu(out)
        else:
            out = out
        if self.InShape is None:
            self.InShape = inputs.size(-2), inputs.size(-1)
            self.OutShape = out.size(-2), out.size(-1)
        if self.maxpool:
            out = self.maxpool(out)
        return out


class ResNetBasicblock(nn.Module):
    expansion = 1
    num_conv = 2

    def __init__(self, inplanes, planes, stride):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ConvBNReLU(inplanes, planes, 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_b = ConvBNReLU(planes, planes, 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=False)
        if stride == 2:
            self.downsample = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=True, has_bn=True, has_relu=False)
        elif inplanes != planes:
            self.downsample = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        else:
            self.downsample = None
        self.out_dim = planes
        self.search_mode = 'basic'

    def get_range(self):
        return self.conv_a.get_range() + self.conv_b.get_range()

    def get_flops(self, channels):
        assert len(channels) == 3, 'invalid channels : {:}'.format(channels)
        flop_A = self.conv_a.get_flops([channels[0], channels[1]])
        flop_B = self.conv_b.get_flops([channels[1], channels[2]])
        if hasattr(self.downsample, 'get_flops'):
            flop_C = self.downsample.get_flops([channels[0], channels[-1]])
        else:
            flop_C = 0
        if channels[0] != channels[-1] and self.downsample is None:
            flop_C = channels[0] * channels[-1] * self.conv_b.OutShape[0] * self.conv_b.OutShape[1]
        return flop_A + flop_B + flop_C

    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

    def search_forward(self, tuple_inputs):
        assert isinstance(tuple_inputs, tuple) and len(tuple_inputs) == 5, 'invalid type input : {:}'.format(type(tuple_inputs))
        inputs, expected_inC, probability, indexes, probs = tuple_inputs
        assert indexes.size(0) == 2 and probs.size(0) == 2 and probability.size(0) == 2
        out_a, expected_inC_a, expected_flop_a = self.conv_a((inputs, expected_inC, probability[0], indexes[0], probs[0]))
        out_b, expected_inC_b, expected_flop_b = self.conv_b((out_a, expected_inC_a, probability[1], indexes[1], probs[1]))
        if self.downsample is not None:
            residual, _, expected_flop_c = self.downsample((inputs, expected_inC, probability[1], indexes[1], probs[1]))
        else:
            residual, expected_flop_c = inputs, 0
        out = additive_func(residual, out_b)
        return nn.functional.relu(out, inplace=True), expected_inC_b, sum([expected_flop_a, expected_flop_b, expected_flop_c])

    def basic_forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = additive_func(residual, basicblock)
        return nn.functional.relu(out, inplace=True)


class ResNetBottleneck(nn.Module):
    expansion = 4
    num_conv = 3

    def __init__(self, inplanes, planes, stride):
        super(ResNetBottleneck, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_1x1 = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_3x3 = ConvBNReLU(planes, planes, 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
        self.conv_1x4 = ConvBNReLU(planes, planes * self.expansion, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        if stride == 2:
            self.downsample = ConvBNReLU(inplanes, planes * self.expansion, 1, 1, 0, False, has_avg=True, has_bn=True, has_relu=False)
        elif inplanes != planes * self.expansion:
            self.downsample = ConvBNReLU(inplanes, planes * self.expansion, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        else:
            self.downsample = None
        self.out_dim = planes * self.expansion
        self.search_mode = 'basic'

    def get_range(self):
        return self.conv_1x1.get_range() + self.conv_3x3.get_range() + self.conv_1x4.get_range()

    def get_flops(self, channels):
        assert len(channels) == 4, 'invalid channels : {:}'.format(channels)
        flop_A = self.conv_1x1.get_flops([channels[0], channels[1]])
        flop_B = self.conv_3x3.get_flops([channels[1], channels[2]])
        flop_C = self.conv_1x4.get_flops([channels[2], channels[3]])
        if hasattr(self.downsample, 'get_flops'):
            flop_D = self.downsample.get_flops([channels[0], channels[-1]])
        else:
            flop_D = 0
        if channels[0] != channels[-1] and self.downsample is None:
            flop_D = channels[0] * channels[-1] * self.conv_1x4.OutShape[0] * self.conv_1x4.OutShape[1]
        return flop_A + flop_B + flop_C + flop_D

    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

    def basic_forward(self, inputs):
        bottleneck = self.conv_1x1(inputs)
        bottleneck = self.conv_3x3(bottleneck)
        bottleneck = self.conv_1x4(bottleneck)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = additive_func(residual, bottleneck)
        return nn.functional.relu(out, inplace=True)

    def search_forward(self, tuple_inputs):
        assert isinstance(tuple_inputs, tuple) and len(tuple_inputs) == 5, 'invalid type input : {:}'.format(type(tuple_inputs))
        inputs, expected_inC, probability, indexes, probs = tuple_inputs
        assert indexes.size(0) == 3 and probs.size(0) == 3 and probability.size(0) == 3
        out_1x1, expected_inC_1x1, expected_flop_1x1 = self.conv_1x1((inputs, expected_inC, probability[0], indexes[0], probs[0]))
        out_3x3, expected_inC_3x3, expected_flop_3x3 = self.conv_3x3((out_1x1, expected_inC_1x1, probability[1], indexes[1], probs[1]))
        out_1x4, expected_inC_1x4, expected_flop_1x4 = self.conv_1x4((out_3x3, expected_inC_3x3, probability[2], indexes[2], probs[2]))
        if self.downsample is not None:
            residual, _, expected_flop_c = self.downsample((inputs, expected_inC, probability[2], indexes[2], probs[2]))
        else:
            residual, expected_flop_c = inputs, 0
        out = additive_func(residual, out_1x4)
        return nn.functional.relu(out, inplace=True), expected_inC_1x4, sum([expected_flop_1x1, expected_flop_3x3, expected_flop_1x4, expected_flop_c])


class SearchShapeImagenetResNet(nn.Module):

    def __init__(self, block_name, layers, deep_stem, num_classes):
        super(SearchShapeImagenetResNet, self).__init__()
        if block_name == 'BasicBlock':
            block = ResNetBasicblock
        elif block_name == 'Bottleneck':
            block = ResNetBottleneck
        else:
            raise ValueError('invalid block : {:}'.format(block_name))
        self.message = 'SearchShapeCifarResNet : Depth : {:} , Layers for each block : {:}'.format(sum(layers) * block.num_conv, layers)
        self.num_classes = num_classes
        if not deep_stem:
            self.layers = nn.ModuleList([ConvBNReLU(3, 64, 7, 2, 3, False, has_avg=False, has_bn=True, has_relu=True, last_max_pool=True)])
            self.channels = [64]
        else:
            self.layers = nn.ModuleList([ConvBNReLU(3, 32, 3, 2, 1, False, has_avg=False, has_bn=True, has_relu=True), ConvBNReLU(32, 64, 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=True, last_max_pool=True)])
            self.channels = [32, 64]
        meta_depth_info = get_depth_choices(layers)
        self.InShape = None
        self.depth_info = OrderedDict()
        self.depth_at_i = OrderedDict()
        for stage, layer_blocks in enumerate(layers):
            cur_block_choices = meta_depth_info[stage]
            assert cur_block_choices[-1] == layer_blocks, 'stage={:}, {:} vs {:}'.format(stage, cur_block_choices, layer_blocks)
            block_choices, xstart = [], len(self.layers)
            for iL in range(layer_blocks):
                iC = self.channels[-1]
                planes = 64 * 2 ** stage
                stride = 2 if stage > 0 and iL == 0 else 1
                module = block(iC, planes, stride)
                self.channels.append(module.out_dim)
                self.layers.append(module)
                self.message += '\nstage={:}, ilayer={:02d}/{:02d}, block={:03d}, iC={:3d}, oC={:3d}, stride={:}'.format(stage, iL, layer_blocks, len(self.layers) - 1, iC, module.out_dim, stride)
                layer_index = len(self.layers) - 1
                if iL + 1 in cur_block_choices:
                    block_choices.append(layer_index)
                if iL + 1 == layer_blocks:
                    self.depth_info[layer_index] = {'choices': block_choices, 'stage': stage, 'xstart': xstart}
        self.depth_info_list = []
        for xend, info in self.depth_info.items():
            self.depth_info_list.append((xend, info))
            xstart, xstage = info['xstart'], info['stage']
            for ilayer in range(xstart, xend + 1):
                idx = bisect_right(info['choices'], ilayer - 1)
                self.depth_at_i[ilayer] = xstage, idx
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(module.out_dim, num_classes)
        self.InShape = None
        self.tau = -1
        self.search_mode = 'basic'
        self.Ranges = []
        self.layer2indexRange = []
        for i, layer in enumerate(self.layers):
            start_index = len(self.Ranges)
            self.Ranges += layer.get_range()
            self.layer2indexRange.append((start_index, len(self.Ranges)))
        self.register_parameter('width_attentions', nn.Parameter(torch.Tensor(len(self.Ranges), get_width_choices(None))))
        self.register_parameter('depth_attentions', nn.Parameter(torch.Tensor(len(layers), meta_depth_info['num'])))
        nn.init.normal_(self.width_attentions, 0, 0.01)
        nn.init.normal_(self.depth_attentions, 0, 0.01)
        self.apply(initialize_resnet)

    def arch_parameters(self, LR=None):
        if LR is None:
            return [self.width_attentions, self.depth_attentions]
        else:
            return [{'params': self.width_attentions, 'lr': LR}, {'params': self.depth_attentions, 'lr': LR}]

    def base_parameters(self):
        return list(self.layers.parameters()) + list(self.avgpool.parameters()) + list(self.classifier.parameters())

    def get_flop(self, mode, config_dict, extra_info):
        if config_dict is not None:
            config_dict = config_dict.copy()
        channels = [3]
        for i, weight in enumerate(self.width_attentions):
            if mode == 'genotype':
                with torch.no_grad():
                    probe = nn.functional.softmax(weight, dim=0)
                    C = self.Ranges[i][torch.argmax(probe).item()]
            else:
                raise ValueError('invalid mode : {:}'.format(mode))
            channels.append(C)
        if mode == 'genotype':
            with torch.no_grad():
                depth_probs = nn.functional.softmax(self.depth_attentions, dim=1)
                choices = torch.argmax(depth_probs, dim=1).cpu().tolist()
        else:
            raise ValueError('invalid mode : {:}'.format(mode))
        selected_layers = []
        for choice, xvalue in zip(choices, self.depth_info_list):
            xtemp = xvalue[1]['choices'][choice] - xvalue[1]['xstart'] + 1
            selected_layers.append(xtemp)
        flop = 0
        for i, layer in enumerate(self.layers):
            s, e = self.layer2indexRange[i]
            xchl = tuple(channels[s:e + 1])
            if i in self.depth_at_i:
                xstagei, xatti = self.depth_at_i[i]
                if xatti <= choices[xstagei]:
                    flop += layer.get_flops(xchl)
                else:
                    flop += 0
            else:
                flop += layer.get_flops(xchl)
        flop += channels[-1] * self.classifier.out_features
        if config_dict is None:
            return flop / 1000000.0
        else:
            config_dict['xchannels'] = channels
            config_dict['xblocks'] = selected_layers
            config_dict['super_type'] = 'infer-shape'
            config_dict['estimated_FLOP'] = flop / 1000000.0
            return flop / 1000000.0, config_dict

    def get_arch_info(self):
        string = 'for depth and width, there are {:} + {:} attention probabilities.'.format(len(self.depth_attentions), len(self.width_attentions))
        string += '\n{:}'.format(self.depth_info)
        discrepancy = []
        with torch.no_grad():
            for i, att in enumerate(self.depth_attentions):
                prob = nn.functional.softmax(att, dim=0)
                prob = prob.cpu()
                selc = prob.argmax().item()
                prob = prob.tolist()
                prob = ['{:.3f}'.format(x) for x in prob]
                xstring = '{:03d}/{:03d}-th : {:}'.format(i, len(self.depth_attentions), ' '.join(prob))
                logt = ['{:.4f}'.format(x) for x in att.cpu().tolist()]
                xstring += '  ||  {:17s}'.format(' '.join(logt))
                prob = sorted([float(x) for x in prob])
                disc = prob[-1] - prob[-2]
                xstring += '  || discrepancy={:.2f} || select={:}/{:}'.format(disc, selc, len(prob))
                discrepancy.append(disc)
                string += '\n{:}'.format(xstring)
            string += '\n-----------------------------------------------'
            for i, att in enumerate(self.width_attentions):
                prob = nn.functional.softmax(att, dim=0)
                prob = prob.cpu()
                selc = prob.argmax().item()
                prob = prob.tolist()
                prob = ['{:.3f}'.format(x) for x in prob]
                xstring = '{:03d}/{:03d}-th : {:}'.format(i, len(self.width_attentions), ' '.join(prob))
                logt = ['{:.3f}'.format(x) for x in att.cpu().tolist()]
                xstring += '  ||  {:52s}'.format(' '.join(logt))
                prob = sorted([float(x) for x in prob])
                disc = prob[-1] - prob[-2]
                xstring += '  || dis={:.2f} || select={:}/{:}'.format(disc, selc, len(prob))
                discrepancy.append(disc)
                string += '\n{:}'.format(xstring)
        return string, discrepancy

    def set_tau(self, tau_max, tau_min, epoch_ratio):
        assert epoch_ratio >= 0 and epoch_ratio <= 1, 'invalid epoch-ratio : {:}'.format(epoch_ratio)
        tau = tau_min + (tau_max - tau_min) * (1 + math.cos(math.pi * epoch_ratio)) / 2
        self.tau = tau

    def get_message(self):
        return self.message

    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

    def search_forward(self, inputs):
        flop_width_probs = nn.functional.softmax(self.width_attentions, dim=1)
        flop_depth_probs = nn.functional.softmax(self.depth_attentions, dim=1)
        flop_depth_probs = torch.flip(torch.cumsum(torch.flip(flop_depth_probs, [1]), 1), [1])
        selected_widths, selected_width_probs = select2withP(self.width_attentions, self.tau)
        selected_depth_probs = select2withP(self.depth_attentions, self.tau, True)
        with torch.no_grad():
            selected_widths = selected_widths.cpu()
        x, last_channel_idx, expected_inC, flops = inputs, 0, 3, []
        feature_maps = []
        for i, layer in enumerate(self.layers):
            selected_w_index = selected_widths[last_channel_idx:last_channel_idx + layer.num_conv]
            selected_w_probs = selected_width_probs[last_channel_idx:last_channel_idx + layer.num_conv]
            layer_prob = flop_width_probs[last_channel_idx:last_channel_idx + layer.num_conv]
            x, expected_inC, expected_flop = layer((x, expected_inC, layer_prob, selected_w_index, selected_w_probs))
            feature_maps.append(x)
            last_channel_idx += layer.num_conv
            if i in self.depth_info:
                choices = self.depth_info[i]['choices']
                xstagei = self.depth_info[i]['stage']
                possible_tensors = []
                max_C = max(feature_maps[A].size(1) for A in choices)
                for tempi, A in enumerate(choices):
                    xtensor = ChannelWiseInter(feature_maps[A], max_C)
                    possible_tensors.append(xtensor)
                weighted_sum = sum(xtensor * W for xtensor, W in zip(possible_tensors, selected_depth_probs[xstagei]))
                x = weighted_sum
            if i in self.depth_at_i:
                xstagei, xatti = self.depth_at_i[i]
                x_expected_flop = flop_depth_probs[xstagei, xatti] * expected_flop
            else:
                x_expected_flop = expected_flop
            flops.append(x_expected_flop)
        flops.append(expected_inC * (self.classifier.out_features * 1.0 / 1000000.0))
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = linear_forward(features, self.classifier)
        return logits, torch.stack([sum(flops)])

    def basic_forward(self, inputs):
        if self.InShape is None:
            self.InShape = inputs.size(-2), inputs.size(-1)
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return features, logits


class ConvBNReLU(nn.Module):
    num_conv = 1

    def __init__(self, nIn, nOut, kernel, stride, padding, bias, has_avg, has_bn, has_relu):
        super(ConvBNReLU, self).__init__()
        self.InShape = None
        self.OutShape = None
        self.choices = get_choices(nOut)
        self.register_buffer('choices_tensor', torch.Tensor(self.choices))
        if has_avg:
            self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.avg = None
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride, padding=padding, dilation=1, groups=1, bias=bias)
        self.has_bn = has_bn
        self.BNs = nn.ModuleList()
        for i, _out in enumerate(self.choices):
            self.BNs.append(nn.BatchNorm2d(_out))
        if has_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None
        self.in_dim = nIn
        self.out_dim = nOut
        self.search_mode = 'basic'

    def get_flops(self, channels, check_range=True, divide=1):
        iC, oC = channels
        if check_range:
            assert iC <= self.conv.in_channels and oC <= self.conv.out_channels, '{:} vs {:}  |  {:} vs {:}'.format(iC, self.conv.in_channels, oC, self.conv.out_channels)
        assert isinstance(self.InShape, tuple) and len(self.InShape) == 2, 'invalid in-shape : {:}'.format(self.InShape)
        assert isinstance(self.OutShape, tuple) and len(self.OutShape) == 2, 'invalid out-shape : {:}'.format(self.OutShape)
        conv_per_position_flops = self.conv.kernel_size[0] * self.conv.kernel_size[1] * 1.0 / self.conv.groups
        all_positions = self.OutShape[0] * self.OutShape[1]
        flops = conv_per_position_flops * all_positions / divide * iC * oC
        if self.conv.bias is not None:
            flops += all_positions / divide
        return flops

    def get_range(self):
        return [self.choices]

    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

    def search_forward(self, tuple_inputs):
        assert isinstance(tuple_inputs, tuple) and len(tuple_inputs) == 5, 'invalid type input : {:}'.format(type(tuple_inputs))
        inputs, expected_inC, probability, index, prob = tuple_inputs
        index, prob = torch.squeeze(index).tolist(), torch.squeeze(prob)
        probability = torch.squeeze(probability)
        assert len(index) == 2, 'invalid length : {:}'.format(index)
        expected_outC = (self.choices_tensor * probability).sum()
        expected_flop = self.get_flops([expected_inC, expected_outC], False, 1000000.0)
        if self.avg:
            out = self.avg(inputs)
        else:
            out = inputs
        out_convs = conv_forward(out, self.conv, [self.choices[i] for i in index])
        out_bns = [self.BNs[idx](out_conv) for idx, out_conv in zip(index, out_convs)]
        out_channel = max([x.size(1) for x in out_bns])
        outA = ChannelWiseInter(out_bns[0], out_channel)
        outB = ChannelWiseInter(out_bns[1], out_channel)
        out = outA * prob[0] + outB * prob[1]
        if self.relu:
            out = self.relu(out)
        else:
            out = out
        return out, expected_outC, expected_flop

    def basic_forward(self, inputs):
        if self.avg:
            out = self.avg(inputs)
        else:
            out = inputs
        conv = self.conv(out)
        if self.has_bn:
            out = self.BNs[-1](conv)
        else:
            out = conv
        if self.relu:
            out = self.relu(out)
        else:
            out = out
        if self.InShape is None:
            self.InShape = inputs.size(-2), inputs.size(-1)
            self.OutShape = out.size(-2), out.size(-1)
        return out


class SimBlock(nn.Module):
    expansion = 1
    num_conv = 1

    def __init__(self, inplanes, planes, stride):
        super(SimBlock, self).__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv = ConvBNReLU(inplanes, planes, 3, stride, 1, False, has_avg=False, has_bn=True, has_relu=True)
        if stride == 2:
            self.downsample = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=True, has_bn=False, has_relu=False)
        elif inplanes != planes:
            self.downsample = ConvBNReLU(inplanes, planes, 1, 1, 0, False, has_avg=False, has_bn=True, has_relu=False)
        else:
            self.downsample = None
        self.out_dim = planes
        self.search_mode = 'basic'

    def get_range(self):
        return self.conv.get_range()

    def get_flops(self, channels):
        assert len(channels) == 2, 'invalid channels : {:}'.format(channels)
        flop_A = self.conv.get_flops([channels[0], channels[1]])
        if hasattr(self.downsample, 'get_flops'):
            flop_C = self.downsample.get_flops([channels[0], channels[-1]])
        else:
            flop_C = 0
        if channels[0] != channels[-1] and self.downsample is None:
            flop_C = channels[0] * channels[-1] * self.conv.OutShape[0] * self.conv.OutShape[1]
        return flop_A + flop_C

    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

    def search_forward(self, tuple_inputs):
        assert isinstance(tuple_inputs, tuple) and len(tuple_inputs) == 5, 'invalid type input : {:}'.format(type(tuple_inputs))
        inputs, expected_inC, probability, indexes, probs = tuple_inputs
        assert indexes.size(0) == 1 and probs.size(0) == 1 and probability.size(0) == 1, 'invalid size : {:}, {:}, {:}'.format(indexes.size(), probs.size(), probability.size())
        out, expected_next_inC, expected_flop = self.conv((inputs, expected_inC, probability[0], indexes[0], probs[0]))
        if self.downsample is not None:
            residual, _, expected_flop_c = self.downsample((inputs, expected_inC, probability[-1], indexes[-1], probs[-1]))
        else:
            residual, expected_flop_c = inputs, 0
        out = additive_func(residual, out)
        return nn.functional.relu(out, inplace=True), expected_next_inC, sum([expected_flop, expected_flop_c])

    def basic_forward(self, inputs):
        basicblock = self.conv(inputs)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        out = additive_func(residual, basicblock)
        return nn.functional.relu(out, inplace=True)


class SearchWidthSimResNet(nn.Module):

    def __init__(self, depth, num_classes):
        super(SearchWidthSimResNet, self).__init__()
        assert (depth - 2) % 3 == 0, 'depth should be one of 5, 8, 11, 14, ... instead of {:}'.format(depth)
        layer_blocks = (depth - 2) // 3
        self.message = 'SearchWidthSimResNet : Depth : {:} , Layers for each block : {:}'.format(depth, layer_blocks)
        self.num_classes = num_classes
        self.channels = [16]
        self.layers = nn.ModuleList([ConvBNReLU(3, 16, 3, 1, 1, False, has_avg=False, has_bn=True, has_relu=True)])
        self.InShape = None
        for stage in range(3):
            for iL in range(layer_blocks):
                iC = self.channels[-1]
                planes = 16 * 2 ** stage
                stride = 2 if stage > 0 and iL == 0 else 1
                module = SimBlock(iC, planes, stride)
                self.channels.append(module.out_dim)
                self.layers.append(module)
                self.message += '\nstage={:}, ilayer={:02d}/{:02d}, block={:03d}, iC={:3d}, oC={:3d}, stride={:}'.format(stage, iL, layer_blocks, len(self.layers) - 1, iC, module.out_dim, stride)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(module.out_dim, num_classes)
        self.InShape = None
        self.tau = -1
        self.search_mode = 'basic'
        self.Ranges = []
        self.layer2indexRange = []
        for i, layer in enumerate(self.layers):
            start_index = len(self.Ranges)
            self.Ranges += layer.get_range()
            self.layer2indexRange.append((start_index, len(self.Ranges)))
        assert len(self.Ranges) + 1 == depth, 'invalid depth check {:} vs {:}'.format(len(self.Ranges) + 1, depth)
        self.register_parameter('width_attentions', nn.Parameter(torch.Tensor(len(self.Ranges), get_choices(None))))
        nn.init.normal_(self.width_attentions, 0, 0.01)
        self.apply(initialize_resnet)

    def arch_parameters(self):
        return [self.width_attentions]

    def base_parameters(self):
        return list(self.layers.parameters()) + list(self.avgpool.parameters()) + list(self.classifier.parameters())

    def get_flop(self, mode, config_dict, extra_info):
        if config_dict is not None:
            config_dict = config_dict.copy()
        channels = [3]
        for i, weight in enumerate(self.width_attentions):
            if mode == 'genotype':
                with torch.no_grad():
                    probe = nn.functional.softmax(weight, dim=0)
                    C = self.Ranges[i][torch.argmax(probe).item()]
            elif mode == 'max':
                C = self.Ranges[i][-1]
            elif mode == 'fix':
                C = int(math.sqrt(extra_info) * self.Ranges[i][-1])
            elif mode == 'random':
                assert isinstance(extra_info, float), 'invalid extra_info : {:}'.format(extra_info)
                with torch.no_grad():
                    prob = nn.functional.softmax(weight, dim=0)
                    approximate_C = int(math.sqrt(extra_info) * self.Ranges[i][-1])
                    for j in range(prob.size(0)):
                        prob[j] = 1 / (abs(j - (approximate_C - self.Ranges[i][j])) + 0.2)
                    C = self.Ranges[i][torch.multinomial(prob, 1, False).item()]
            else:
                raise ValueError('invalid mode : {:}'.format(mode))
            channels.append(C)
        flop = 0
        for i, layer in enumerate(self.layers):
            s, e = self.layer2indexRange[i]
            xchl = tuple(channels[s:e + 1])
            flop += layer.get_flops(xchl)
        flop += channels[-1] * self.classifier.out_features
        if config_dict is None:
            return flop / 1000000.0
        else:
            config_dict['xchannels'] = channels
            config_dict['super_type'] = 'infer-width'
            config_dict['estimated_FLOP'] = flop / 1000000.0
            return flop / 1000000.0, config_dict

    def get_arch_info(self):
        string = 'for width, there are {:} attention probabilities.'.format(len(self.width_attentions))
        discrepancy = []
        with torch.no_grad():
            for i, att in enumerate(self.width_attentions):
                prob = nn.functional.softmax(att, dim=0)
                prob = prob.cpu()
                selc = prob.argmax().item()
                prob = prob.tolist()
                prob = ['{:.3f}'.format(x) for x in prob]
                xstring = '{:03d}/{:03d}-th : {:}'.format(i, len(self.width_attentions), ' '.join(prob))
                logt = ['{:.3f}'.format(x) for x in att.cpu().tolist()]
                xstring += '  ||  {:52s}'.format(' '.join(logt))
                prob = sorted([float(x) for x in prob])
                disc = prob[-1] - prob[-2]
                xstring += '  || dis={:.2f} || select={:}/{:}'.format(disc, selc, len(prob))
                discrepancy.append(disc)
                string += '\n{:}'.format(xstring)
        return string, discrepancy

    def set_tau(self, tau_max, tau_min, epoch_ratio):
        assert epoch_ratio >= 0 and epoch_ratio <= 1, 'invalid epoch-ratio : {:}'.format(epoch_ratio)
        tau = tau_min + (tau_max - tau_min) * (1 + math.cos(math.pi * epoch_ratio)) / 2
        self.tau = tau

    def get_message(self):
        return self.message

    def forward(self, inputs):
        if self.search_mode == 'basic':
            return self.basic_forward(inputs)
        elif self.search_mode == 'search':
            return self.search_forward(inputs)
        else:
            raise ValueError('invalid search_mode = {:}'.format(self.search_mode))

    def search_forward(self, inputs):
        flop_probs = nn.functional.softmax(self.width_attentions, dim=1)
        selected_widths, selected_probs = select2withP(self.width_attentions, self.tau)
        with torch.no_grad():
            selected_widths = selected_widths.cpu()
        x, last_channel_idx, expected_inC, flops = inputs, 0, 3, []
        for i, layer in enumerate(self.layers):
            selected_w_index = selected_widths[last_channel_idx:last_channel_idx + layer.num_conv]
            selected_w_probs = selected_probs[last_channel_idx:last_channel_idx + layer.num_conv]
            layer_prob = flop_probs[last_channel_idx:last_channel_idx + layer.num_conv]
            x, expected_inC, expected_flop = layer((x, expected_inC, layer_prob, selected_w_index, selected_w_probs))
            last_channel_idx += layer.num_conv
            flops.append(expected_flop)
        flops.append(expected_inC * (self.classifier.out_features * 1.0 / 1000000.0))
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = linear_forward(features, self.classifier)
        return logits, torch.stack([sum(flops)])

    def basic_forward(self, inputs):
        if self.InShape is None:
            self.InShape = inputs.size(-2), inputs.size(-1)
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
        features = self.avgpool(x)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return features, logits


class NetworkCIFAR(nn.Module):

    def __init__(self, C, N, stem_multiplier, auxiliary, genotype, num_classes):
        super(NetworkCIFAR, self).__init__()
        self._C = C
        self._layerN = N
        self._stem_multiplier = stem_multiplier
        C_curr = self._stem_multiplier * C
        self.stem = CifarHEAD(C_curr)
        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
        block_indexs = [0] * N + [-1] + [1] * N + [-1] + [2] * N
        block2index = {(0): [], (1): [], (2): []}
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        reduction_prev, spatial, dims = False, 1, []
        self.auxiliary_index = None
        self.auxiliary_head = None
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            cell = InferCell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell._multiplier * C_curr
            if reduction and C_curr == C * 4:
                if auxiliary:
                    self.auxiliary_head = AuxiliaryHeadCIFAR(C_prev, num_classes)
                    self.auxiliary_index = index
            if reduction:
                spatial *= 2
            dims.append((C_prev, spatial))
        self._Layer = len(self.cells)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.drop_path_prob = -1

    def update_drop_path(self, drop_path_prob):
        self.drop_path_prob = drop_path_prob

    def auxiliary_param(self):
        if self.auxiliary_head is None:
            return []
        else:
            return list(self.auxiliary_head.parameters())

    def get_message(self):
        return self.extra_repr()

    def extra_repr(self):
        return '{name}(C={_C}, N={_layerN}, L={_Layer}, stem={_stem_multiplier}, drop-path={drop_path_prob})'.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs):
        stem_feature, logits_aux = self.stem(inputs), None
        cell_results = [stem_feature, stem_feature]
        for i, cell in enumerate(self.cells):
            cell_feature = cell(cell_results[-2], cell_results[-1], self.drop_path_prob)
            cell_results.append(cell_feature)
            if self.auxiliary_index is not None and i == self.auxiliary_index and self.training:
                logits_aux = self.auxiliary_head(cell_results[-1])
        out = self.global_pooling(cell_results[-1])
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        if logits_aux is None:
            return out, logits
        else:
            return out, [logits, logits_aux]


class NetworkImageNet(nn.Module):

    def __init__(self, C, N, auxiliary, genotype, num_classes):
        super(NetworkImageNet, self).__init__()
        self._C = C
        self._layerN = N
        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
        self.stem0 = nn.Sequential(nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(C // 2), nn.ReLU(inplace=True), nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(C))
        self.stem1 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(C))
        C_prev_prev, C_prev, C_curr, reduction_prev = C, C, C, True
        self.cells = nn.ModuleList()
        self.auxiliary_index = None
        for i, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            cell = InferCell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell._multiplier * C_curr
            if reduction and C_curr == C * 4:
                C_to_auxiliary = C_prev
                self.auxiliary_index = i
        self._NNN = len(self.cells)
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        else:
            self.auxiliary_head = None
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.drop_path_prob = -1

    def update_drop_path(self, drop_path_prob):
        self.drop_path_prob = drop_path_prob

    def extra_repr(self):
        return '{name}(C={_C}, N=[{_layerN}, {_NNN}], aux-index={auxiliary_index}, drop-path={drop_path_prob})'.format(name=self.__class__.__name__, **self.__dict__)

    def get_message(self):
        return self.extra_repr()

    def auxiliary_param(self):
        if self.auxiliary_head is None:
            return []
        else:
            return list(self.auxiliary_head.parameters())

    def forward(self, inputs):
        s0 = self.stem0(inputs)
        s1 = self.stem1(s0)
        logits_aux = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == self.auxiliary_index and self.auxiliary_head and self.training:
                logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if logits_aux is None:
            return out, logits
        else:
            return out, [logits, logits_aux]


class MixedOp(nn.Module):

    def __init__(self, C, stride, PRIMITIVES):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.name2idx = {}
        for idx, primitive in enumerate(PRIMITIVES):
            op = OPS[primitive](C, C, stride, False)
            self._ops.append(op)
            assert primitive not in self.name2idx, '{:} has already in'.format(primitive)
            self.name2idx[primitive] = idx

    def forward(self, x, weights, op_name):
        if op_name is None:
            if weights is None:
                return [op(x) for op in self._ops]
            else:
                return sum(w * op(x) for w, op in zip(weights, self._ops))
        else:
            op_index = self.name2idx[op_name]
            return self._ops[op_index](x)


class SearchCell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, PRIMITIVES, use_residual):
        super(SearchCell, self).__init__()
        self.reduction = reduction
        self.PRIMITIVES = deepcopy(PRIMITIVES)
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, 2, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier
        self._use_residual = use_residual
        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, self.PRIMITIVES)
                self._ops.append(op)

    def extra_repr(self):
        return '{name}(residual={_use_residual}, steps={_steps}, multiplier={_multiplier})'.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, S0, S1, weights, connect, adjacency, drop_prob, modes):
        if modes[0] is None:
            if modes[1] == 'normal':
                output = self.__forwardBoth(S0, S1, weights, connect, adjacency, drop_prob)
            elif modes[1] == 'only_W':
                output = self.__forwardOnlyW(S0, S1, drop_prob)
        else:
            test_genotype = modes[0]
            if self.reduction:
                operations, concats = test_genotype.reduce, test_genotype.reduce_concat
            else:
                operations, concats = test_genotype.normal, test_genotype.normal_concat
            s0, s1 = self.preprocess0(S0), self.preprocess1(S1)
            states, offset = [s0, s1], 0
            assert self._steps == len(operations), '{:} vs. {:}'.format(self._steps, len(operations))
            for i, (opA, opB) in enumerate(operations):
                A = self._ops[offset + opA[1]](states[opA[1]], None, opA[0])
                B = self._ops[offset + opB[1]](states[opB[1]], None, opB[0])
                state = A + B
                offset += len(states)
                states.append(state)
            output = torch.cat([states[i] for i in concats], dim=1)
        if self._use_residual and S1.size() == output.size():
            return S1 + output
        else:
            return output

    def __forwardBoth(self, S0, S1, weights, connect, adjacency, drop_prob):
        s0, s1 = self.preprocess0(S0), self.preprocess1(S1)
        states, offset = [s0, s1], 0
        for i in range(self._steps):
            clist = []
            for j, h in enumerate(states):
                x = self._ops[offset + j](h, weights[offset + j], None)
                if self.training and drop_prob > 0.0:
                    x = drop_path(x, math.pow(drop_prob, 1.0 / len(states)))
                clist.append(x)
            connection = torch.mm(connect['{:}'.format(i)], adjacency[i]).squeeze(0)
            state = sum(w * node for w, node in zip(connection, clist))
            offset += len(states)
            states.append(state)
        return torch.cat(states[-self._multiplier:], dim=1)

    def __forwardOnlyW(self, S0, S1, drop_prob):
        s0, s1 = self.preprocess0(S0), self.preprocess1(S1)
        states, offset = [s0, s1], 0
        for i in range(self._steps):
            clist = []
            for j, h in enumerate(states):
                xs = self._ops[offset + j](h, None, None)
                clist += xs
            if self.training and drop_prob > 0.0:
                xlist = [drop_path(x, math.pow(drop_prob, 1.0 / len(states))) for x in clist]
            else:
                xlist = clist
            state = sum(xlist) * 2 / len(xlist)
            offset += len(states)
            states.append(state)
        return torch.cat(states[-self._multiplier:], dim=1)


class InferCell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(InferCell, self).__init__()
        None
        if reduction_prev is None:
            self.preprocess0 = Identity()
        elif reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, 2)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        if reduction:
            step_ops, concat = genotype.reduce, genotype.reduce_concat
        else:
            step_ops, concat = genotype.normal, genotype.normal_concat
        self._steps = len(step_ops)
        self._concat = concat
        self._multiplier = len(concat)
        self._ops = nn.ModuleList()
        self._indices = []
        for operations in step_ops:
            for name, index in operations:
                stride = 2 if reduction and index < 2 else 1
                if reduction_prev is None and index == 0:
                    op = OPS[name](C_prev_prev, C, stride, True)
                else:
                    op = OPS[name](C, C, stride, True)
                self._ops.append(op)
                self._indices.append(index)

    def extra_repr(self):
        return '{name}(steps={_steps}, concat={_concat})'.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, S0, S1, drop_prob):
        s0 = self.preprocess0(S0)
        s1 = self.preprocess1(S1)
        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.0:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            state = h1 + h2
            states += [state]
        output = torch.cat([states[i] for i in self._concat], dim=1)
        return output


class ImageNetHEAD(nn.Sequential):

    def __init__(self, C, stride=2):
        super(ImageNetHEAD, self).__init__()
        self.add_module('conv1', nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False))
        self.add_module('bn1', nn.BatchNorm2d(C // 2))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(C // 2, C, kernel_size=3, stride=stride, padding=1, bias=False))
        self.add_module('bn2', nn.BatchNorm2d(C))


class CifarHEAD(nn.Sequential):

    def __init__(self, C):
        super(CifarHEAD, self).__init__()
        self.add_module('conv', nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False))
        self.add_module('bn', nn.BatchNorm2d(C))


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), nn.Conv2d(C, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 768, 2, bias=False), nn.BatchNorm2d(768), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False), nn.Conv2d(C, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 768, 2, bias=False), nn.BatchNorm2d(768), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class POOLING(nn.Module):

    def __init__(self, C_in, C_out, stride, mode):
        super(POOLING, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0)
        if mode == 'avg':
            self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == 'max':
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)

    def forward(self, inputs):
        if self.preprocess is not None:
            x = self.preprocess(inputs)
        else:
            x = inputs
        return self.op(x)


class Conv313(nn.Module):

    def __init__(self, C_in, C_out, stride, affine):
        super(Conv313, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_out, (1, 3), stride=(1, stride), padding=(0, 1), bias=False), nn.Conv2d(C_out, C_out, (3, 1), stride=(stride, 1), padding=(1, 0), bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class Conv717(nn.Module):

    def __init__(self, C_in, C_out, stride, affine):
        super(Conv717, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_out, (1, 7), stride=(1, stride), padding=(0, 3), bias=False), nn.Conv2d(C_out, C_out, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False), nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_in, affine=affine), nn.ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, ::self.stride, ::self.stride].mul(0.0)

    def extra_repr(self):
        return 'stride={stride}'.format(**self.__dict__)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, stride, affine=True):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False))
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        elif stride == 4:
            assert C_out % 4 == 0, 'C_out : {:}'.format(C_out)
            self.convs = nn.ModuleList()
            for i in range(4):
                self.convs.append(nn.Conv2d(C_in, C_out // 4, 1, stride=stride, padding=0, bias=False))
            self.pad = nn.ConstantPad2d((0, 3, 0, 3), 0)
        else:
            raise ValueError('Invalid stride : {:}'.format(stride))
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        y = self.pad(x)
        if self.stride == 2:
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        else:
            out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:-2, 1:-2]), self.convs[2](y[:, :, 2:-1, 2:-1]), self.convs[3](y[:, :, 3:, 3:])], dim=1)
        out = self.bn(out)
        return out

    def extra_repr(self):
        return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AuxiliaryHeadCIFAR,
     lambda: ([], {'C': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 8, 8])], {}),
     True),
    (AuxiliaryHeadImageNet,
     lambda: ([], {'C': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 8, 8])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CifarHEAD,
     lambda: ([], {'C': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Controller,
     lambda: ([], {'num_edge': 4, 'num_ops': 4}),
     lambda: ([], {}),
     False),
    (Conv313,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'stride': 1, 'affine': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv717,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'stride': 1, 'affine': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CrossEntropyLabelSmooth,
     lambda: ([], {'num_classes': 4, 'epsilon': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.zeros([4, 4, 4], dtype=torch.int64)], {}),
     True),
    (DenseNet,
     lambda: ([], {'growthRate': 4, 'depth': 1, 'reduction': 4, 'nClasses': 4, 'bottleneck': 4}),
     lambda: ([torch.rand([4, 3, 32, 32])], {}),
     True),
    (DilConv,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ImageNetHEAD,
     lambda: ([], {'C': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (NAS201SearchCell,
     lambda: ([], {'C_in': [4, 4], 'C_out': [4, 4], 'stride': [4, 4], 'max_nodes': 1, 'op_names': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Policy,
     lambda: ([], {'max_nodes': 4, 'search_space': [4, 4]}),
     lambda: ([], {}),
     True),
    (ReLUConvBN,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SepConv,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SingleLayer,
     lambda: ([], {'nChannels': 4, 'growthRate': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transition,
     lambda: ([], {'nChannels': 4, 'nOutChannels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WideBasicblock,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Zero,
     lambda: ([], {'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_D_X_Y_AutoDL_Projects(_paritybench_base):
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

