import sys
_module = sys.modules[__name__]
del sys
dataset = _module
main = _module
resnext = _module
sknet = _module
train = _module

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


import numpy as np


from torch.utils.data import Dataset


import torch


from torch import nn


from torch import optim


class ResNeXtUnit(nn.Module):

    def __init__(self, in_features, out_features, mid_features=None, stride=1, groups=32):
        super(ResNeXtUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features / 2)
        self.feas = nn.Sequential(nn.Conv2d(in_features, mid_features, 1, stride=1), nn.BatchNorm2d(mid_features), nn.Conv2d(mid_features, mid_features, 3, stride=stride, padding=1, groups=groups), nn.BatchNorm2d(mid_features), nn.Conv2d(mid_features, out_features, 1, stride=1), nn.BatchNorm2d(out_features))
        if in_features == out_features:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(in_features, out_features, 1, stride=stride), nn.BatchNorm2d(out_features))

    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)


class ResNeXt(nn.Module):

    def __init__(self, class_num):
        super(ResNeXt, self).__init__()
        self.basic_conv = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64))
        self.stage_1 = nn.Sequential(ResNeXtUnit(64, 256, mid_features=128), nn.ReLU(), ResNeXtUnit(256, 256), nn.ReLU(), ResNeXtUnit(256, 256), nn.ReLU())
        self.stage_2 = nn.Sequential(ResNeXtUnit(256, 512, stride=2), nn.ReLU(), ResNeXtUnit(512, 512), nn.ReLU(), ResNeXtUnit(512, 512), nn.ReLU())
        self.stage_3 = nn.Sequential(ResNeXtUnit(512, 1024, stride=2), nn.ReLU(), ResNeXtUnit(1024, 1024), nn.ReLU(), ResNeXtUnit(1024, 1024), nn.ReLU())
        self.pool = nn.AvgPool2d(8)
        self.classifier = nn.Sequential(nn.Linear(1024, class_num))

    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.pool(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea


class SKConv(nn.Module):

    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G), nn.BatchNorm2d(features), nn.ReLU(inplace=False)))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class SKUnit(nn.Module):

    def __init__(self, in_features, out_features, WH, M, G, r, mid_features=None, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features / 2)
        self.feas = nn.Sequential(nn.Conv2d(in_features, mid_features, 1, stride=1), nn.BatchNorm2d(mid_features), SKConv(mid_features, WH, M, G, r, stride=stride, L=L), nn.BatchNorm2d(mid_features), nn.Conv2d(mid_features, out_features, 1, stride=1), nn.BatchNorm2d(out_features))
        if in_features == out_features:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(in_features, out_features, 1, stride=stride), nn.BatchNorm2d(out_features))

    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)


class SKNet(nn.Module):

    def __init__(self, class_num):
        super(SKNet, self).__init__()
        self.basic_conv = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64))
        self.stage_1 = nn.Sequential(SKUnit(64, 256, 32, 2, 8, 2, stride=2), nn.ReLU(), SKUnit(256, 256, 32, 2, 8, 2), nn.ReLU(), SKUnit(256, 256, 32, 2, 8, 2), nn.ReLU())
        self.stage_2 = nn.Sequential(SKUnit(256, 512, 32, 2, 8, 2, stride=2), nn.ReLU(), SKUnit(512, 512, 32, 2, 8, 2), nn.ReLU(), SKUnit(512, 512, 32, 2, 8, 2), nn.ReLU())
        self.stage_3 = nn.Sequential(SKUnit(512, 1024, 32, 2, 8, 2, stride=2), nn.ReLU(), SKUnit(1024, 1024, 32, 2, 8, 2), nn.ReLU(), SKUnit(1024, 1024, 32, 2, 8, 2), nn.ReLU())
        self.pool = nn.AvgPool2d(8)
        self.classifier = nn.Sequential(nn.Linear(1024, class_num))

    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.pool(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ResNeXtUnit,
     lambda: ([], {'in_features': 64, 'out_features': 64}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (SKConv,
     lambda: ([], {'features': 4, 'WH': 4, 'M': 4, 'G': 4, 'r': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SKNet,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_pppLang_SKNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

