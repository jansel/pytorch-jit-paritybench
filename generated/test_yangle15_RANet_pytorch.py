import sys
_module = sys.modules[__name__]
del sys
adaptive_inference = _module
args = _module
dataloader = _module
main = _module
RANet = _module
models = _module
op_counter = _module

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


import math


import torch


import torch.nn as nn


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import time


import torch.optim


import torch.backends.cudnn as cudnn


import copy


import numpy as np


import torch.nn.functional as F


from torch.autograd import Variable


from functools import reduce


class ConvBasic(nn.Module):

    def __init__(self, nIn, nOut, kernel=3, stride=1, padding=1):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(nOut), nn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class ConvBN(nn.Module):

    def __init__(self, nIn, nOut, type: str, bnAfter, bnWidth):
        """
        a basic conv in RANet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bnAfter: the location of batch Norm
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bnAfter is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.Conv2d(nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))
            if type == 'normal':
                layer.append(nn.Conv2d(nInner, nOut, kernel_size=3, stride=1, padding=1, bias=False))
            elif type == 'down':
                layer.append(nn.Conv2d(nInner, nOut, kernel_size=3, stride=2, padding=1, bias=False))
            else:
                raise ValueError
            layer.append(nn.BatchNorm2d(nOut))
            layer.append(nn.ReLU(True))
        else:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.BatchNorm2d(nIn))
            layer.append(nn.ReLU(True))
            layer.append(nn.Conv2d(nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))
            if type == 'normal':
                layer.append(nn.Conv2d(nInner, nOut, kernel_size=3, stride=1, padding=1, bias=False))
            elif type == 'down':
                layer.append(nn.Conv2d(nInner, nOut, kernel_size=3, stride=2, padding=1, bias=False))
            else:
                raise ValueError
        self.net = nn.Sequential(*layer)

    def forward(self, x):
        return self.net(x)


class ConvUpNormal(nn.Module):

    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2, compress_factor, down_sample):
        """
        The convolution with normal and up-sampling connection.
        """
        super(ConvUpNormal, self).__init__()
        self.conv_up = ConvBN(nIn2, math.floor(nOut * compress_factor), 'normal', bottleneck, bnWidth2)
        if down_sample:
            self.conv_normal = ConvBN(nIn1, nOut - math.floor(nOut * compress_factor), 'down', bottleneck, bnWidth1)
        else:
            self.conv_normal = ConvBN(nIn1, nOut - math.floor(nOut * compress_factor), 'normal', bottleneck, bnWidth1)

    def forward(self, x):
        res = self.conv_normal(x[1])
        _, _, h, w = res.size()
        res = [F.interpolate(x[1], size=(h, w), mode='bilinear', align_corners=True), F.interpolate(self.conv_up(x[0]), size=(h, w), mode='bilinear', align_corners=True), res]
        return torch.cat(res, dim=1)


class ConvNormal(nn.Module):

    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        """
        The convolution with normal connection.
        """
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, 'normal', bottleneck, bnWidth)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        res = [x[0], self.conv_normal(x[0])]
        return torch.cat(res, dim=1)


class _BlockNormal(nn.Module):

    def __init__(self, num_layers, nIn, growth_rate, reduction_rate, trans, bnFactor):
        """
        The basic computational block in RANet with num_layers layers.
        trans: If True, the block will add a transiation layer at the end of the block
                with reduction_rate.
        """
        super(_BlockNormal, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            self.layers.append(ConvNormal(nIn + i * growth_rate, growth_rate, True, bnFactor))
        nOut = nIn + num_layers * growth_rate
        self.trans_flag = trans
        if trans:
            self.trans = ConvBasic(nOut, math.floor(1.0 * reduction_rate * nOut), kernel=1, stride=1, padding=0)

    def forward(self, x):
        output = [x]
        for i in range(self.num_layers):
            x = self.layers[i](x)
            output.append(x)
        x = output[-1]
        if self.trans_flag:
            x = self.trans(x)
        return x, output

    def _blockType(self):
        return 'norm'


class _BlockUpNormal(nn.Module):

    def __init__(self, num_layers, nIn, nIn_lowFtrs, growth_rate, reduction_rate, trans, down, compress_factor, bnFactor1, bnFactor2):
        """
        The basic fusion block in RANet with num_layers layers.
        trans: If True, the block will add a transiation layer at the end of the block
                with reduction_rate.
        compress_factor: There will be compress_factor*100% information from the previous
                sub-network.  
        """
        super(_BlockUpNormal, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers - 1):
            self.layers.append(ConvUpNormal(nIn + i * growth_rate, nIn_lowFtrs[i], growth_rate, True, bnFactor1, bnFactor2, compress_factor, False))
        self.layers.append(ConvUpNormal(nIn + (i + 1) * growth_rate, nIn_lowFtrs[i + 1], growth_rate, True, bnFactor1, bnFactor2, compress_factor, down))
        nOut = nIn + num_layers * growth_rate
        self.conv_last = ConvBasic(nIn_lowFtrs[num_layers], math.floor(nOut * compress_factor), kernel=1, stride=1, padding=0)
        nOut = nOut + math.floor(nOut * compress_factor)
        self.trans_flag = trans
        if trans:
            self.trans = ConvBasic(nOut, math.floor(1.0 * reduction_rate * nOut), kernel=1, stride=1, padding=0)

    def forward(self, x, low_feat):
        output = [x]
        for i in range(self.num_layers):
            inp = [low_feat[i]]
            inp.append(x)
            x = self.layers[i](inp)
            output.append(x)
        x = output[-1]
        _, _, h, w = x.size()
        x = [x]
        x.append(F.interpolate(self.conv_last(low_feat[self.num_layers]), size=(h, w), mode='bilinear', align_corners=True))
        x = torch.cat(x, dim=1)
        if self.trans_flag:
            x = self.trans(x)
        return x, output

    def _blockType(self):
        return 'up'


class RAFirstLayer(nn.Module):

    def __init__(self, nIn, nOut, args):
        """
        RAFirstLayer gennerates the base features for RANet.
        The scale 1 means the lowest resoultion in the network.
        """
        super(RAFirstLayer, self).__init__()
        _grFactor = args.grFactor[::-1]
        _scale_list = args.scale_list[::-1]
        self.layers = nn.ModuleList()
        if args.data.startswith('cifar'):
            self.layers.append(ConvBasic(nIn, nOut * _grFactor[0], kernel=3, stride=1, padding=1))
        elif args.data == 'ImageNet':
            conv = nn.Sequential(nn.Conv2d(nIn, nOut * _grFactor[0], 7, 2, 3), nn.BatchNorm2d(nOut * _grFactor[0]), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
            self.layers.append(conv)
        nIn = nOut * _grFactor[0]
        s = _scale_list[0]
        for i in range(1, args.nScales):
            if s == _scale_list[i]:
                self.layers.append(ConvBasic(nIn, nOut * _grFactor[i], kernel=3, stride=1, padding=1))
            else:
                self.layers.append(ConvBasic(nIn, nOut * _grFactor[i], kernel=3, stride=2, padding=1))
                s = _scale_list[i]
            nIn = nOut * _grFactor[i]

    def forward(self, x):
        res = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            res.append(x)
        return res[::-1]


class ClassifierModule(nn.Module):

    def __init__(self, m, channel, num_classes):
        super(ClassifierModule, self).__init__()
        self.m = m
        self.linear = nn.Linear(channel, num_classes)

    def forward(self, x):
        res = self.m(x)
        res = res.view(res.size(0), -1)
        return self.linear(res)


class RANet(nn.Module):

    def __init__(self, args):
        super(RANet, self).__init__()
        self.scale_flows = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.compress_factor = args.compress_factor
        self.bnFactor = copy.copy(args.bnFactor)
        scale_list = args.scale_list
        self.nScales = len(args.scale_list)
        self.nBlocks = [0]
        for i in range(self.nScales):
            self.nBlocks.append(args.block_step * i + args.nBlocks)
        self.steps = args.step
        self.FirstLayer = RAFirstLayer(3, args.nChannels, args)
        steps = [args.step]
        for ii in range(self.nScales):
            scale_flow = nn.ModuleList()
            n_block_curr = 1
            nIn = args.nChannels * args.grFactor[ii]
            _nIn_lowFtrs = []
            for i in range(self.nBlocks[ii + 1]):
                growth_rate = args.growthRate * args.grFactor[ii]
                trans = self._trans_flag(n_block_curr, n_block_all=self.nBlocks[ii + 1], inScale=scale_list[ii])
                if n_block_curr > self.nBlocks[ii]:
                    m, nOuts = self._build_norm_block(nIn, steps[n_block_curr - 1], growth_rate, args.reduction, trans, bnFactor=self.bnFactor[ii])
                    if args.stepmode == 'even':
                        steps.append(args.step)
                    elif args.stepmode == 'lg':
                        steps.append(steps[-1] + args.step)
                    else:
                        raise NotImplementedError
                elif n_block_curr in self.nBlocks[:ii + 1][-(scale_list[ii] - 1):]:
                    m, nOuts = self._build_upNorm_block(nIn, nIn_lowFtrs[i], steps[n_block_curr - 1], growth_rate, args.reduction, trans, down=True, bnFactor1=self.bnFactor[ii], bnFactor2=self.bnFactor[ii - 1])
                else:
                    m, nOuts = self._build_upNorm_block(nIn, nIn_lowFtrs[i], steps[n_block_curr - 1], growth_rate, args.reduction, trans, down=False, bnFactor1=self.bnFactor[ii], bnFactor2=self.bnFactor[ii - 1])
                nIn = nOuts[-1]
                scale_flow.append(m)
                if n_block_curr > self.nBlocks[ii]:
                    if args.data.startswith('cifar100'):
                        self.classifier.append(self._build_classifier_cifar(nIn, 100))
                    elif args.data.startswith('cifar10'):
                        self.classifier.append(self._build_classifier_cifar(nIn, 10))
                    elif args.data == 'ImageNet':
                        self.classifier.append(self._build_classifier_imagenet(nIn, 1000))
                    else:
                        raise NotImplementedError
                _nIn_lowFtrs.append(nOuts[:-1])
                n_block_curr += 1
            nIn_lowFtrs = _nIn_lowFtrs
            self.scale_flows.append(scale_flow)
        args.num_exits = len(self.classifier)
        for m in self.scale_flows:
            for _m in m.modules():
                self._init_weights(_m)
        for m in self.classifier:
            for _m in m.modules():
                self._init_weights(_m)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _build_norm_block(self, nIn, step, growth_rate, reduction_rate, trans, bnFactor=2):
        block = _BlockNormal(step, nIn, growth_rate, reduction_rate, trans, bnFactor=bnFactor)
        nOuts = []
        for i in range(step + 1):
            nOut = nIn + i * growth_rate
            nOuts.append(nOut)
        if trans:
            nOut = math.floor(1.0 * reduction_rate * nOut)
        nOuts.append(nOut)
        return block, nOuts

    def _build_upNorm_block(self, nIn, nIn_lowFtr, step, growth_rate, reduction_rate, trans, down, bnFactor1=1, bnFactor2=2):
        compress_factor = self.compress_factor
        block = _BlockUpNormal(step, nIn, nIn_lowFtr, growth_rate, reduction_rate, trans, down, compress_factor, bnFactor1=bnFactor1, bnFactor2=bnFactor2)
        nOuts = []
        for i in range(step + 1):
            nOut = nIn + i * growth_rate
            nOuts.append(nOut)
        nOut = nOut + math.floor(nOut * compress_factor)
        if trans:
            nOut = math.floor(1.0 * reduction_rate * nOut)
        nOuts.append(nOut)
        return block, nOuts

    def _trans_flag(self, n_block_curr, n_block_all, inScale):
        flag = False
        for i in range(inScale - 1):
            if n_block_curr == math.floor((i + 1) * n_block_all / inScale):
                flag = True
        return flag

    def forward(self, x):
        inp = self.FirstLayer(x)
        res, low_ftrs = [], []
        classifier_idx = 0
        for ii in range(self.nScales):
            _x = inp[ii]
            _low_ftrs = []
            n_block_curr = 0
            for i in range(self.nBlocks[ii + 1]):
                if self.scale_flows[ii][i]._blockType() == 'norm':
                    _x, _low_ftr = self.scale_flows[ii][i](_x)
                    _low_ftrs.append(_low_ftr)
                else:
                    _x, _low_ftr = self.scale_flows[ii][i](_x, low_ftrs[i])
                    _low_ftrs.append(_low_ftr)
                n_block_curr += 1
                if n_block_curr > self.nBlocks[ii]:
                    res.append(self.classifier[classifier_idx](_x))
                    classifier_idx += 1
            low_ftrs = _low_ftrs
        return res

    def _build_classifier_cifar(self, nIn, num_classes):
        interChannels1, interChannels2 = 128, 128
        conv = nn.Sequential(ConvBasic(nIn, interChannels1, kernel=3, stride=2, padding=1), ConvBasic(interChannels1, interChannels2, kernel=3, stride=2, padding=1), nn.AvgPool2d(2))
        return ClassifierModule(conv, interChannels2, num_classes)

    def _build_classifier_imagenet(self, nIn, num_classes):
        conv = nn.Sequential(ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1), ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1), nn.AvgPool2d(2))
        return ClassifierModule(conv, nIn, num_classes)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ClassifierModule,
     lambda: ([], {'m': _mock_layer(), 'channel': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (ConvBasic,
     lambda: ([], {'nIn': 4, 'nOut': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvNormal,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'bottleneck': 4, 'bnWidth': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_BlockNormal,
     lambda: ([], {'num_layers': 1, 'nIn': 4, 'growth_rate': 4, 'reduction_rate': 4, 'trans': 4, 'bnFactor': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_yangle15_RANet_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

