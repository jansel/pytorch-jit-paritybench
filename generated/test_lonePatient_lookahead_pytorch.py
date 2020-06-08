import sys
_module = sys.modules[__name__]
del sys
nn = _module
optimizer = _module
progressbar = _module
run = _module
tools = _module
trainingmonitor = _module

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


import torch.nn.functional as F


from torch.utils.data import DataLoader


import torch.optim as optim


class ResidualBlock(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel,
            kernel_size=3, stride=stride, padding=1, bias=False), nn.
            BatchNorm2d(outchannel), nn.ReLU(inplace=True), nn.Conv2d(
            outchannel, outchannel, kernel_size=3, stride=1, padding=1,
            bias=False), nn.BatchNorm2d(outchannel))
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel,
                kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(
                outchannel))

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1,
            padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_lonePatient_lookahead_pytorch(_paritybench_base):
    pass

    def test_000(self):
        self._check(ResidualBlock(*[], **{'inchannel': 4, 'outchannel': 4}), [torch.rand([4, 4, 4, 4])], {})
