import sys
_module = sys.modules[__name__]
del sys
data_loader = _module
evaluation = _module
hg = _module
models = _module
test = _module
train = _module
visualize = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import copy


import torch


from torch.autograd import Variable


import torch.nn as nn


from torch.nn import UpsamplingNearest2d


from torch.nn import Upsample


import torch.optim as optim


from torch.backends import cudnn


class Residual(nn.Module):
    """
    残差模块，并不改变特征图的宽高
    """

    def __init__(self, ins, outs):
        super(Residual, self).__init__()
        self.convBlock = nn.Sequential(nn.BatchNorm2d(ins), nn.ReLU(inplace=True), nn.Conv2d(ins, outs / 2, 1), nn.BatchNorm2d(outs / 2), nn.ReLU(inplace=True), nn.Conv2d(outs / 2, outs / 2, 3, 1, 1), nn.BatchNorm2d(outs / 2), nn.ReLU(inplace=True), nn.Conv2d(outs / 2, outs, 1))
        if ins != outs:
            self.skipConv = nn.Conv2d(ins, outs, 1)
        self.ins = ins
        self.outs = outs

    def forward(self, x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        return x


class HourGlass(nn.Module):
    """不改变特征图的高宽"""

    def __init__(self, n=4, f=128):
        """
        :param n: hourglass模块的层级数目
        :param f: hourglass模块中的特征图数量
        :return:
        """
        super(HourGlass, self).__init__()
        self._n = n
        self._f = f
        self._init_layers(self._n, self._f)

    def _init_layers(self, n, f):
        setattr(self, 'res' + str(n) + '_1', Residual(f, f))
        setattr(self, 'pool' + str(n) + '_1', nn.MaxPool2d(2, 2))
        setattr(self, 'res' + str(n) + '_2', Residual(f, f))
        if n > 1:
            self._init_layers(n - 1, f)
        else:
            self.res_center = Residual(f, f)
        setattr(self, 'res' + str(n) + '_3', Residual(f, f))
        setattr(self, 'unsample' + str(n), Upsample(scale_factor=2))

    def _forward(self, x, n, f):
        up1 = x
        up1 = eval('self.res' + str(n) + '_1')(up1)
        low1 = eval('self.pool' + str(n) + '_1')(x)
        low1 = eval('self.res' + str(n) + '_2')(low1)
        if n > 1:
            low2 = self._forward(low1, n - 1, f)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.' + 'res' + str(n) + '_3')(low3)
        up2 = eval('self.' + 'unsample' + str(n)).forward(low3)
        return up1 + up2

    def forward(self, x):
        return self._forward(x, self._n, self._f)


class Lin(nn.Module):

    def __init__(self, numIn=128, numout=15):
        super(Lin, self).__init__()
        self.conv = nn.Conv2d(numIn, numout, 1)
        self.bn = nn.BatchNorm2d(numout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class StackedHourGlass(nn.Module):

    def __init__(self, nFeats=256, nStack=8, nJoints=18):
        """
        输入： 256^2
        """
        super(StackedHourGlass, self).__init__()
        self._nFeats = nFeats
        self._nStack = nStack
        self._nJoints = nJoints
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.res1 = Residual(64, 128)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.res2 = Residual(128, 128)
        self.res3 = Residual(128, self._nFeats)
        self._init_stacked_hourglass()

    def _init_stacked_hourglass(self):
        for i in range(self._nStack):
            setattr(self, 'hg' + str(i), HourGlass(4, self._nFeats))
            setattr(self, 'hg' + str(i) + '_res1', Residual(self._nFeats, self._nFeats))
            setattr(self, 'hg' + str(i) + '_lin1', Lin(self._nFeats, self._nFeats))
            setattr(self, 'hg' + str(i) + '_conv_pred', nn.Conv2d(self._nFeats, self._nJoints, 1))
            if i < self._nStack - 1:
                setattr(self, 'hg' + str(i) + '_conv1', nn.Conv2d(self._nFeats, self._nFeats, 1))
                setattr(self, 'hg' + str(i) + '_conv2', nn.Conv2d(self._nJoints, self._nFeats, 1))

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.pool1(x)
        x = self.res2(x)
        x = self.res3(x)
        out = []
        inter = x
        for i in range(self._nStack):
            hg = eval('self.hg' + str(i))(inter)
            ll = hg
            ll = eval('self.hg' + str(i) + '_res1')(ll)
            ll = eval('self.hg' + str(i) + '_lin1')(ll)
            tmpOut = eval('self.hg' + str(i) + '_conv_pred')(ll)
            out.append(tmpOut)
            if i < self._nStack - 1:
                ll_ = eval('self.hg' + str(i) + '_conv1')(ll)
                tmpOut_ = eval('self.hg' + str(i) + '_conv2')(tmpOut)
                inter = inter + ll_ + tmpOut_
        return out


class KFSGNet(nn.Module):

    def __init__(self):
        super(KFSGNet, self).__init__()
        self.__conv1 = nn.Conv2d(1, 64, 1)
        self.__relu1 = nn.ReLU(inplace=True)
        self.__conv2 = nn.Conv2d(64, 128, 1)
        self.__relu2 = nn.ReLU(inplace=True)
        self.__hg = HourGlass()
        self.__lin = Lin()

    def forward(self, x):
        x = self.__relu1(self.__conv1(x))
        x = self.__relu2(self.__conv2(x))
        x = self.__hg(x)
        x = self.__lin(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Lin,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 64, 64])], {}),
     True),
]

class Test_raymon_tian_hourglass_facekeypoints_detection(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

