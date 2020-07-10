import sys
_module = sys.modules[__name__]
del sys
RandAugment = _module
augmentations = _module
common = _module
data = _module
imagenet = _module
lr_scheduler = _module
metrics = _module
networks = _module
pyramidnet = _module
resnet = _module
shakedrop = _module
shakeshake = _module
shake_resnet = _module
shake_resnext = _module
shakeshake = _module
wideresnet = _module
smooth_ce = _module
train = _module
setup = _module

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
xrange = range
wraps = functools.wraps


import random


import numpy as np


import torch


import logging


import torchvision


from torch.utils.data import SubsetRandomSampler


from torch.utils.data import Sampler


from torch.utils.data.dataset import ConcatDataset


from torchvision.transforms import transforms


from sklearn.model_selection import StratifiedShuffleSplit


from torchvision.datasets.utils import check_integrity


from torchvision.datasets.utils import download_url


import copy


from collections import defaultdict


from torch import nn


from torch.nn import DataParallel


import torch.backends.cudnn as cudnn


import torch.nn as nn


import math


import torch.nn.functional as F


from torch.autograd import Variable


import torch.nn.init as init


from torch.nn.modules.module import Module


import itertools


from collections import OrderedDict


from torch import optim


from torch.nn.parallel.data_parallel import DataParallel


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
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


class PyramidNet(nn.Module):

    def __init__(self, dataset, depth, alpha, num_classes, bottleneck=True):
        super(PyramidNet, self).__init__()
        self.dataset = dataset
        if self.dataset.startswith('cifar'):
            self.inplanes = 16
            if bottleneck:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock
            self.addrate = alpha / (3 * n * 1.0)
            self.ps_shakedrop = [(1.0 - (1.0 - 0.5 / (3 * n) * (i + 1))) for i in range(3 * n)]
            self.input_featuremap_dim = self.inplanes
            self.conv1 = nn.Conv2d(3, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)
            self.featuremap_dim = self.input_featuremap_dim
            self.layer1 = self.pyramidal_make_layer(block, n)
            self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
            self.layer3 = self.pyramidal_make_layer(block, n, stride=2)
            self.final_featuremap_dim = self.input_featuremap_dim
            self.bn_final = nn.BatchNorm2d(self.final_featuremap_dim)
            self.relu_final = nn.ReLU(inplace=True)
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(self.final_featuremap_dim, num_classes)
        elif dataset == 'imagenet':
            blocks = {(18): BasicBlock, (34): BasicBlock, (50): Bottleneck, (101): Bottleneck, (152): Bottleneck, (200): Bottleneck}
            layers = {(18): [2, 2, 2, 2], (34): [3, 4, 6, 3], (50): [3, 4, 6, 3], (101): [3, 4, 23, 3], (152): [3, 8, 36, 3], (200): [3, 24, 36, 3]}
            if layers.get(depth) is None:
                if bottleneck == True:
                    blocks[depth] = Bottleneck
                    temp_cfg = int((depth - 2) / 12)
                else:
                    blocks[depth] = BasicBlock
                    temp_cfg = int((depth - 2) / 8)
                layers[depth] = [temp_cfg, temp_cfg, temp_cfg, temp_cfg]
                None
            self.inplanes = 64
            self.addrate = alpha / (sum(layers[depth]) * 1.0)
            self.input_featuremap_dim = self.inplanes
            self.conv1 = nn.Conv2d(3, self.input_featuremap_dim, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.featuremap_dim = self.input_featuremap_dim
            self.layer1 = self.pyramidal_make_layer(blocks[depth], layers[depth][0])
            self.layer2 = self.pyramidal_make_layer(blocks[depth], layers[depth][1], stride=2)
            self.layer3 = self.pyramidal_make_layer(blocks[depth], layers[depth][2], stride=2)
            self.layer4 = self.pyramidal_make_layer(blocks[depth], layers[depth][3], stride=2)
            self.final_featuremap_dim = self.input_featuremap_dim
            self.bn_final = nn.BatchNorm2d(self.final_featuremap_dim)
            self.relu_final = nn.ReLU(inplace=True)
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(self.final_featuremap_dim, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        assert len(self.ps_shakedrop) == 0, self.ps_shakedrop

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)
        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample, p_shakedrop=self.ps_shakedrop.pop(0)))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1, p_shakedrop=self.ps_shakedrop.pop(0)))
            self.featuremap_dim = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.bn_final(x)
            x = self.relu_final(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        elif self.dataset == 'imagenet':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn_final(x)
            x = self.relu_final(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x


class ResNet(nn.Module):

    def __init__(self, dataset, depth, num_classes, bottleneck=False):
        super(ResNet, self).__init__()
        self.dataset = dataset
        if self.dataset.startswith('cifar'):
            self.inplanes = 16
            None
            if bottleneck == True:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 16, n)
            self.layer2 = self._make_layer(block, 32, n, stride=2)
            self.layer3 = self._make_layer(block, 64, n, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64 * block.expansion, num_classes)
        elif dataset == 'imagenet':
            blocks = {(18): BasicBlock, (34): BasicBlock, (50): Bottleneck, (101): Bottleneck, (152): Bottleneck, (200): Bottleneck}
            layers = {(18): [2, 2, 2, 2], (34): [3, 4, 6, 3], (50): [3, 4, 6, 3], (101): [3, 4, 23, 3], (152): [3, 8, 36, 3], (200): [3, 24, 36, 3]}
            assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
            self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
            self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
            self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)
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
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        elif self.dataset == 'imagenet':
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


class ShakeDropFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpha_range=[-1, 1]):
        if training:
            gate = torch.FloatTensor([0]).bernoulli_(1 - p_drop)
            ctx.save_for_backward(gate)
            if gate.item() == 0:
                alpha = torch.FloatTensor(x.size(0)).uniform_(*alpha_range)
                alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x)
                return alpha * x
            else:
                return x
        else:
            return (1 - p_drop) * x

    @staticmethod
    def backward(ctx, grad_output):
        gate = ctx.saved_tensors[0]
        if gate.item() == 0:
            beta = torch.FloatTensor(grad_output.size(0)).uniform_(0, 1)
            beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
            beta = Variable(beta)
            return beta * grad_output, None, None, None
        else:
            return grad_output, None, None, None


class ShakeDrop(nn.Module):

    def __init__(self, p_drop=0.5, alpha_range=[-1, 1]):
        super(ShakeDrop, self).__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range

    def forward(self, x):
        return ShakeDropFunction.apply(x, self.training, self.p_drop, self.alpha_range)


class ShakeShake(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x1, x2, training=True):
        if training:
            alpha = torch.FloatTensor(x1.size(0)).uniform_()
            alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x1)
        else:
            alpha = 0.5
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_output):
        beta = torch.FloatTensor(grad_output.size(0)).uniform_()
        beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
        beta = Variable(beta)
        return beta * grad_output, (1 - beta) * grad_output, None


class Shortcut(nn.Module):

    def __init__(self, in_ch, out_ch, stride):
        super(Shortcut, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        h = F.relu(x)
        h1 = F.avg_pool2d(h, 1, self.stride)
        h1 = self.conv1(h1)
        h2 = F.avg_pool2d(F.pad(h, (-1, 1, -1, 1)), 1, self.stride)
        h2 = self.conv2(h2)
        h = torch.cat((h1, h2), 1)
        return self.bn(h)


class ShakeBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super(ShakeBlock, self).__init__()
        self.equal_io = in_ch == out_ch
        self.shortcut = self.equal_io and None or Shortcut(in_ch, out_ch, stride=stride)
        self.branch1 = self._make_branch(in_ch, out_ch, stride)
        self.branch2 = self._make_branch(in_ch, out_ch, stride)

    def forward(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        h = ShakeShake.apply(h1, h2, self.training)
        h0 = x if self.equal_io else self.shortcut(x)
        return h + h0

    def _make_branch(self, in_ch, out_ch, stride=1):
        return nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=False), nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False), nn.BatchNorm2d(out_ch))


class ShakeResNet(nn.Module):

    def __init__(self, depth, w_base, label):
        super(ShakeResNet, self).__init__()
        n_units = (depth - 2) / 6
        in_chs = [16, w_base, w_base * 2, w_base * 4]
        self.in_chs = in_chs
        self.c_in = nn.Conv2d(3, in_chs[0], 3, padding=1)
        self.layer1 = self._make_layer(n_units, in_chs[0], in_chs[1])
        self.layer2 = self._make_layer(n_units, in_chs[1], in_chs[2], 2)
        self.layer3 = self._make_layer(n_units, in_chs[2], in_chs[3], 2)
        self.fc_out = nn.Linear(in_chs[3], label)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        h = self.c_in(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = F.relu(h)
        h = F.avg_pool2d(h, 8)
        h = h.view(-1, self.in_chs[3])
        h = self.fc_out(h)
        return h

    def _make_layer(self, n_units, in_ch, out_ch, stride=1):
        layers = []
        for i in range(int(n_units)):
            layers.append(ShakeBlock(in_ch, out_ch, stride=stride))
            in_ch, stride = out_ch, 1
        return nn.Sequential(*layers)


class ShakeBottleNeck(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch, cardinary, stride=1):
        super(ShakeBottleNeck, self).__init__()
        self.equal_io = in_ch == out_ch
        self.shortcut = None if self.equal_io else Shortcut(in_ch, out_ch, stride=stride)
        self.branch1 = self._make_branch(in_ch, mid_ch, out_ch, cardinary, stride)
        self.branch2 = self._make_branch(in_ch, mid_ch, out_ch, cardinary, stride)

    def forward(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        h = ShakeShake.apply(h1, h2, self.training)
        h0 = x if self.equal_io else self.shortcut(x)
        return h + h0

    def _make_branch(self, in_ch, mid_ch, out_ch, cardinary, stride=1):
        return nn.Sequential(nn.Conv2d(in_ch, mid_ch, 1, padding=0, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=False), nn.Conv2d(mid_ch, mid_ch, 3, padding=1, stride=stride, groups=cardinary, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=False), nn.Conv2d(mid_ch, out_ch, 1, padding=0, bias=False), nn.BatchNorm2d(out_ch))


class ShakeResNeXt(nn.Module):

    def __init__(self, depth, w_base, cardinary, label):
        super(ShakeResNeXt, self).__init__()
        n_units = (depth - 2) // 9
        n_chs = [64, 128, 256, 1024]
        self.n_chs = n_chs
        self.in_ch = n_chs[0]
        self.c_in = nn.Conv2d(3, n_chs[0], 3, padding=1)
        self.layer1 = self._make_layer(n_units, n_chs[0], w_base, cardinary)
        self.layer2 = self._make_layer(n_units, n_chs[1], w_base, cardinary, 2)
        self.layer3 = self._make_layer(n_units, n_chs[2], w_base, cardinary, 2)
        self.fc_out = nn.Linear(n_chs[3], label)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        h = self.c_in(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = F.relu(h)
        h = F.avg_pool2d(h, 8)
        h = h.view(-1, self.n_chs[3])
        h = self.fc_out(h)
        return h

    def _make_layer(self, n_units, n_ch, w_base, cardinary, stride=1):
        layers = []
        mid_ch, out_ch = n_ch * (w_base // 64) * cardinary, n_ch * 4
        for i in range(n_units):
            layers.append(ShakeBottleNeck(self.in_ch, mid_ch, out_ch, cardinary, stride=stride))
            self.in_ch, stride = out_ch, 1
        return nn.Sequential(*layers)


_bn_momentum = 0.1


class WideBasic(nn.Module):

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=_bn_momentum)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes, momentum=_bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True))

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):

    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        assert (depth - 4) % 6 == 0, 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor
        nStages = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(WideBasic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=_bn_momentum)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean
    Examples::
        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)
        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


class SmoothCrossEntropyLoss(Module):

    def __init__(self, label_smoothing=0.0, size_average=True):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.size_average = size_average

    def forward(self, input, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float()
        if self.label_smoothing > 0.0:
            s_by_c = self.label_smoothing / len(input[0])
            smooth = torch.zeros_like(target)
            smooth = smooth + s_by_c
            target = target * (1.0 - s_by_c) + smooth
        return cross_entropy(input, target, self.size_average)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShakeBlock,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ShakeBottleNeck,
     lambda: ([], {'in_ch': 4, 'mid_ch': 4, 'out_ch': 4, 'cardinary': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ShakeDrop,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ShakeResNeXt,
     lambda: ([], {'depth': 1, 'w_base': 4, 'cardinary': 4, 'label': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ShakeResNet,
     lambda: ([], {'depth': 1, 'w_base': 4, 'label': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Shortcut,
     lambda: ([], {'in_ch': 4, 'out_ch': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SmoothCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (WideBasic,
     lambda: ([], {'in_planes': 4, 'planes': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_ildoonet_pytorch_randaugment(_paritybench_base):
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

