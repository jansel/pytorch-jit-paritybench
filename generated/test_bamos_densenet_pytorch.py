import sys
_module = sys.modules[__name__]
del sys
densenet = _module
make_graph = _module
plot = _module
train = _module

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


import torch


import torch.nn as nn


import torch.legacy as legacy


import torch.optim as optim


import torch.nn.functional as F


from torch.autograd import Variable


from torch.utils.data import DataLoader


import math


import numpy as np


class FcCat(nn.Module):

    def __init__(self, nIn, nOut):
        super(FcCat, self).__init__()
        self.fc = nn.Linear(nIn, nOut, bias=False)

    def forward(self, x):
        out = torch.cat((x, self.fc(x)), 1)
        return out


class Net(nn.Module):

    def __init__(self, nFeatures, nHidden1, nHidden2):
        super(Net, self).__init__()
        self.l1 = FcCat(nFeatures, nHidden1)
        self.l2 = FcCat(nFeatures + nHidden1, nHidden2)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        return out


class Bottleneck(nn.Module):

    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
            padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):

    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
            padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):

    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias
            =False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()
        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias
            =False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks,
            bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks,
            bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks,
            bottleneck)
        nChannels += nDenseBlocks * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = F.log_softmax(self.fc(out))
        return out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_bamos_densenet_pytorch(_paritybench_base):
    pass

    def test_000(self):
        self._check(FcCat(*[], **{'nIn': 4, 'nOut': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Net(*[], **{'nFeatures': 4, 'nHidden1': 4, 'nHidden2': 4}), [torch.rand([4, 4])], {})

    def test_002(self):
        self._check(Bottleneck(*[], **{'nChannels': 4, 'growthRate': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(SingleLayer(*[], **{'nChannels': 4, 'growthRate': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Transition(*[], **{'nChannels': 4, 'nOutChannels': 4}), [torch.rand([4, 4, 4, 4])], {})
