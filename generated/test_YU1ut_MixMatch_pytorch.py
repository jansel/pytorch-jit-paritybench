import sys
_module = sys.modules[__name__]
del sys
cifar10 = _module
wideresnet = _module
train = _module
utils = _module
eval = _module
logger = _module
misc = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
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


import random


import numpy as np


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data as data


import torch.nn.init as init


from torch.autograd import Variable


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0,
        activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride
            =stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = not self.equalInOut and nn.Conv2d(in_planes,
            out_planes, kernel_size=1, stride=stride, padding=0, bias=False
            ) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, block, stride,
        dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes,
            nb_layers, stride, dropRate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,
        dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes,
                out_planes, i == 0 and stride or 1, dropRate,
                activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):

    def __init__(self, num_classes, depth=28, widen_factor=2, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 *
            widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
            padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1,
            dropRate, activate_before_residual=True)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2,
            dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2,
            dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_YU1ut_MixMatch_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(BasicBlock(*[], **{'in_planes': 4, 'out_planes': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(WideResNet(*[], **{'num_classes': 4}), [torch.rand([4, 3, 64, 64])], {})

