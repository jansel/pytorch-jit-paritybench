import sys
_module = sys.modules[__name__]
del sys
WRN = _module
densenet = _module
train = _module
utils = _module
vgg = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import math


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import numpy as np


from torch.autograd import Variable


import torchvision.datasets as dset


import torchvision.transforms as transforms


from torch.utils.data import DataLoader


import torchvision.models as models


import logging


from torch.autograd import Variable as V


import time


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, dropRate, layer_index):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride
            =stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = not self.equalInOut and nn.Conv2d(in_planes,
            out_planes, kernel_size=1, stride=stride, padding=0, bias=False
            ) or None
        self.active = True
        self.layer_index = layer_index

    def forward(self, x):
        if not self.active:
            self.eval()
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        out = torch.add(x if self.equalInOut else self.convShortcut(x), out)
        if self.active:
            return out
        else:
            return out.detach()


scale_fn = {'linear': lambda x: x, 'squared': lambda x: x ** 2, 'cubic': lambda
    x: x ** 3}


class DenseNet(nn.Module):

    def __init__(self, growthRate, depth, nClasses, epochs, t_0, scale_lr=
        True, how_scale='cubic', const_time=False, dropRate=0.0):
        super(DenseNet, self).__init__()
        widen_factor = growthRate
        num_classes = nClasses
        self.epochs = epochs
        self.t_0 = t_0
        self.scale_lr = scale_lr
        self.how_scale = how_scale
        self.const_time = const_time
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 *
            widen_factor]
        assert (depth - 4) % 6 == 0
        n = int((depth - 4) / 6)
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
            padding=1, bias=False)
        self.conv1.layer_index = 0
        self.conv1.active = True
        self.layer_index = 1
        self.block1 = self._make_layer(n, nChannels[0], nChannels[1], block,
            1, dropRate)
        self.block2 = self._make_layer(n, nChannels[1], nChannels[2], block,
            2, dropRate)
        self.block3 = self._make_layer(n, nChannels[2], nChannels[3], block,
            2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.bn1.active = True
        self.fc.active = True
        self.bn1.layer_index = self.layer_index
        self.fc.layer_index = self.layer_index
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
            if hasattr(m, 'active'):
                m.lr_ratio = scale_fn[self.how_scale](self.t_0 + (1 - self.
                    t_0) * float(m.layer_index) / self.layer_index)
                m.max_j = self.epochs * 1000 * m.lr_ratio
                m.lr = 0.1 / m.lr_ratio if self.scale_lr else 0.1
        self.optim = optim.SGD([{'params': m.parameters(), 'lr': m.lr,
            'layer_index': m.layer_index} for m in self.modules() if
            hasattr(m, 'active')], nesterov=True, momentum=0.9,
            weight_decay=0.0001)
        self.j = 0
        self.lr_sched = {'itr': 0}

    def _make_layer(self, nb_layers, in_planes, out_planes, block, stride,
        dropRate=0.0):
        layers = []
        None
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes,
                out_planes, i == 0 and stride or 1, dropRate, self.layer_index)
                )
            self.layer_index += 1
        return nn.Sequential(*layers)

    def update_lr(self):
        for m in self.modules():
            if hasattr(m, 'active') and m.active:
                if self.j > m.max_j:
                    m.active = False
                    for i, group in enumerate(self.optim.param_groups):
                        if group['layer_index'] == m.layer_index:
                            self.optim.param_groups.remove(group)
                else:
                    for i, group in enumerate(self.optim.param_groups):
                        if group['layer_index'] == m.layer_index:
                            self.optim.param_groups[i]['lr'
                                ] = 0.05 / m.lr_ratio * (1 + np.cos(np.pi *
                                self.j / m.max_j)
                                ) if self.scale_lr else 0.05 * (1 + np.cos(
                                np.pi * self.j / m.max_j))
        self.j += 1

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return F.log_softmax(self.fc(out))


class Bottleneck(nn.Module):

    def __init__(self, nChannels, growthRate, layer_index):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
            padding=1, bias=False)
        self.active = True
        self.layer_index = layer_index

    def forward(self, x):
        if not self.active:
            self.eval()
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        if self.active:
            return out
        else:
            return out.detach()


class SingleLayer(nn.Module):

    def __init__(self, nChannels, growthRate, layer_index):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
            padding=1, bias=False)
        self.layer_index = layer_index
        self.active = True

    def forward(self, x):
        if not self.active:
            self.eval()
        out = self.conv1(self.bn1(F.relu(x)))
        out = torch.cat((x, out), 1)
        if self.active:
            return out
        else:
            return out.detach()


class Transition(nn.Module):

    def __init__(self, nChannels, nOutChannels, layer_index):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias
            =False)
        self.active = True
        self.layer_index = layer_index

    def forward(self, x):
        if not self.active:
            self.eval()
        out = self.conv1(self.bn1(F.relu(x)))
        out = F.avg_pool2d(out, 2)
        if self.active:
            return out
        else:
            return out.detach()


def calc_speedup(growthRate, nDenseBlocks, t_0, how_scale):
    HW = [32 ** 2, 16 ** 2, 8 ** 2]
    c = [3 * (2 * growthRate) * HW[0] * 9]
    n = 2
    for i in range(3):
        for j in range(nDenseBlocks):
            c.append(n * (4 * growthRate * growthRate) * HW[i] + 4 * 9 *
                growthRate * growthRate * HW[i])
            n += 1
        n = math.floor(n * 0.5)
    C = 2 * sum(c)
    C_f = sum(c) + sum([(c_i * scale_fn[how_scale](t_0 + (1 - t_0) * float(
        index) / len(c))) for index, c_i in enumerate(c)])
    if how_scale == 'linear':
        return 1.3 * (1 - float(C_f) / C)
    else:
        return 1 - float(C_f) / C


class DenseNet(nn.Module):

    def __init__(self, growthRate, depth, nClasses, epochs, t_0, scale_lr=
        True, how_scale='cubic', const_time=False, reduction=0.5,
        bottleneck=True):
        super(DenseNet, self).__init__()
        self.epochs = epochs
        self.t_0 = t_0
        self.scale_lr = scale_lr
        self.how_scale = how_scale
        self.const_time = const_time
        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2
        speedup = calc_speedup(growthRate, nDenseBlocks, t_0, how_scale)
        None
        if self.const_time:
            self.epochs /= 1 - speedup
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias
            =False)
        self.conv1.layer_index = 0
        self.conv1.active = True
        self.layer_index = 1
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks,
            bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels, self.layer_index)
        self.layer_index += 1
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks,
            bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels, self.layer_index)
        self.layer_index += 1
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks,
            bottleneck)
        nChannels += nDenseBlocks * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)
        self.bn1.active = True
        self.fc.active = True
        self.bn1.layer_index = self.layer_index
        self.fc.layer_index = self.layer_index
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
            if hasattr(m, 'active'):
                m.lr_ratio = scale_fn[self.how_scale](self.t_0 + (1 - self.
                    t_0) * float(m.layer_index) / self.layer_index)
                m.max_j = self.epochs * 1000 * m.lr_ratio
                m.lr = 0.1 / m.lr_ratio if self.scale_lr else 0.1
        self.optim = optim.SGD([{'params': m.parameters(), 'lr': m.lr,
            'layer_index': m.layer_index} for m in self.modules() if
            hasattr(m, 'active')], nesterov=True, momentum=0.9,
            weight_decay=0.0001)
        self.j = 0
        self.lr_sched = {'itr': 0}

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, self.
                    layer_index))
            else:
                layers.append(SingleLayer(nChannels, growthRate, self.
                    layer_index))
            nChannels += growthRate
            self.layer_index += 1
        return nn.Sequential(*layers)

    def update_lr(self):
        for m in self.modules():
            if hasattr(m, 'active') and m.active:
                if self.j > m.max_j:
                    m.active = False
                    for i, group in enumerate(self.optim.param_groups):
                        if group['layer_index'] == m.layer_index:
                            self.optim.param_groups.remove(group)
                else:
                    for i, group in enumerate(self.optim.param_groups):
                        if group['layer_index'] == m.layer_index:
                            self.optim.param_groups[i]['lr'
                                ] = 0.05 / m.lr_ratio * (1 + np.cos(np.pi *
                                self.j / m.max_j)
                                ) if self.scale_lr else 0.05 * (1 + np.cos(
                                np.pi * self.j / m.max_j))
        self.j += 1

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = F.log_softmax(self.fc(out))
        return out


class Layer(nn.Module):

    def __init__(self, n_in, n_out, layer_index):
        super(Layer, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, padding=1, bias=
            False)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.layer_index = layer_index
        self.active = True

    def forward(self, x):
        if not self.active:
            self.eval()
        out = F.relu(self.bn1(self.conv1(x)))
        if self.active:
            return out
        else:
            return out.detach()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ajbrock_FreezeOut(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Bottleneck(*[], **{'nChannels': 4, 'growthRate': 4, 'layer_index': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Layer(*[], **{'n_in': 4, 'n_out': 4, 'layer_index': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(SingleLayer(*[], **{'nChannels': 4, 'growthRate': 4, 'layer_index': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(Transition(*[], **{'nChannels': 4, 'nOutChannels': 4, 'layer_index': 1}), [torch.rand([4, 4, 4, 4])], {})

