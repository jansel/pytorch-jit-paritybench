import sys
_module = sys.modules[__name__]
del sys
architect = _module
auto_deeplab = _module
dataloaders = _module
custom_transforms = _module
datasets = _module
cityscapes = _module
coco = _module
combine_dbs = _module
pascal = _module
sbd = _module
utils = _module
genotypes = _module
model_search = _module
mypath = _module
operations = _module
train_autodeeplab = _module

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


import torch


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.data import DataLoader


import random


import scipy.misc as m


from torch.utils import data


from torchvision import transforms


from torch.utils.data import Dataset


import torch.utils.data as data


import scipy.io


from torch.autograd import Variable


class ASPP(nn.Module):

    def __init__(self, in_channels, paddings, dilations, num_classes):
        super(ASPP, self).__init__()
        self._num_classes = num_classes
        self.conv11 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False), nn.BatchNorm2d(in_channels))
        self.conv33 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=paddings, dilation=dilations, bias=False), nn.BatchNorm2d(in_channels))
        self.conv_p = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, bias=False), nn.BatchNorm2d(in_channels))
        self.concate_conv = nn.Sequential(nn.Conv2d(in_channels * 3, self._num_classes, 1, stride=1, padding=0))

    def forward(self, x):
        conv11 = self.conv11(x)
        conv33 = self.conv33(x)
        image_pool = nn.AvgPool2d(kernel_size=x.size()[2:])
        image_pool = image_pool(x)
        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
        upsample = upsample(image_pool)
        upsample = self.conv_p(upsample)
        concate = torch.cat([conv11, conv33, upsample], dim=1)
        return self.concate_conv(concate)


PRIMITIVES = ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, ::self.stride, ::self.stride].mul(0.0)


OPS = {'none': lambda C, stride, affine: Zero(stride), 'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False), 'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1), 'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine), 'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine), 'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine), 'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine), 'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine)}


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class FactorizedIncrease(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(FactorizedIncrease, self).__init__()
        self._in_channel = in_channel
        self.op = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), nn.ReLU(inplace=False), nn.Conv2d(self._in_channel, out_channel, 1, stride=1, padding=0), nn.BatchNorm2d(out_channel))

    def forward(self, x):
        return self.op(x)


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, rate):
        super(Cell, self).__init__()
        self.C_out = C
        if C_prev_prev != -1:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        if rate == 2:
            self.preprocess1 = FactorizedReduce(C_prev, C, affine=False)
        elif rate == 0:
            self.preprocess1 = FactorizedIncrease(C_prev, C)
        else:
            self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier
        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                if C_prev_prev != -1 and j != 0:
                    op = MixedOp(C, stride)
                else:
                    stride = 1
                    op = None
                stride = 1
                self._ops.append(op)
        self.ReLUConvBN = ReLUConvBN(self._multiplier * self.C_out, self.C_out, 1, 1, 0)

    def forward(self, s0, s1, weights):
        if s0 is not None:
            s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states) if h is not None)
            offset += len(states)
            states.append(s)
        concat_feature = torch.cat(states[-self._multiplier:], dim=1)
        return self.ReLUConvBN(concat_feature)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DilConv,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FactorizedIncrease,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FactorizedReduce,
     lambda: ([], {'C_in': 4, 'C_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MixedOp,
     lambda: ([], {'C': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ReLUConvBN,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SepConv,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Zero,
     lambda: ([], {'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_MenghaoGuo_AutoDeeplab(_paritybench_base):
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

