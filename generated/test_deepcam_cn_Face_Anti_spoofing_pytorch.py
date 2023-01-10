import sys
_module = sys.modules[__name__]
del sys
dataset = _module
train = _module

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


import numpy as np


import scipy.io as sio


import torch


from torch.utils import data


import random


import torchvision.transforms as standard_transforms


from torch import optim


from torch import nn


from torch.utils.data import DataLoader


class Net(nn.Module):

    def __init__(self, scale=1.0, expand_ratio=1):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride=1):
            return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.PReLU(oup))

        def conv_dw(inp, oup, stride=1):
            return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.PReLU(inp), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.PReLU(oup))
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout()
        self.head = conv_bn(3, int(32 * scale))
        self.step1 = nn.Sequential(conv_dw(int(32 * scale), int(64 * scale), 2), conv_dw(int(64 * scale), int(128 * scale)), conv_dw(int(128 * scale), int(128 * scale)))
        self.step1_shotcut = conv_dw(int(32 * scale), int(128 * scale), 2)
        self.step2 = nn.Sequential(conv_dw(int(128 * scale), int(128 * scale), 2), conv_dw(int(128 * scale), int(256 * scale)), conv_dw(int(256 * scale), int(256 * scale)))
        self.step2_shotcut = conv_dw(int(128 * scale), int(256 * scale), 2)
        self.depth_ret = nn.Sequential(nn.Conv2d(int(256 * scale), int(256 * scale), 3, 1, 1, groups=int(256 * scale), bias=False), nn.BatchNorm2d(int(256 * scale)), nn.Conv2d(int(256 * scale), 2, 1, 1, 0, bias=False))
        self.depth_shotcut = conv_dw(int(256 * scale), 2)
        self.class_ret = nn.Linear(2048, 2)

    def forward(self, x):
        head = self.head(x)
        step1 = self.step1(head) + self.step1_shotcut(head)
        step2 = self.dropout(self.step2(step1) + self.step2_shotcut(step1))
        depth = self.softmax(self.depth_ret(step2))
        class_pre = self.depth_shotcut(step2) + depth
        class_pre = class_pre.view(-1, 2048)
        class_ret = self.class_ret(class_pre)
        return depth, class_ret


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, eps=1e-07):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class DepthFocalLoss(nn.Module):

    def __init__(self, gamma=1, eps=1e-07):
        super(DepthFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.MSELoss(reduction='none')

    def forward(self, input, target):
        loss = self.ce(input, target)
        loss = loss ** self.gamma
        return loss.mean()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DepthFocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (FocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_deepcam_cn_Face_Anti_spoofing_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

