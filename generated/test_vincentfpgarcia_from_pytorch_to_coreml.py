import sys
_module = sys.modules[__name__]
del sys
model = _module
step1 = _module
step2 = _module
step3 = _module
step4_part1 = _module
step4_part2 = _module
step4_part3 = _module
step4_part4 = _module
step5_part1 = _module
step5_part2 = _module
step5_part3 = _module
step6_part1 = _module
step6_part2 = _module
step6_part3 = _module

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


import torch.nn as nn


import torch


import torchvision


import torchvision.transforms as transforms


import torch.optim as optim


class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(num_features=6), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(num_features=12), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(num_features=24), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc_layers = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(in_features=24 * 4 * 4, out_features=192), nn.ReLU(inplace=True), nn.Linear(in_features=192, out_features=96), nn.ReLU(inplace=True), nn.Linear(in_features=96, out_features=10))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 24 * 4 * 4)
        x = self.fc_layers(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MyNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_vincentfpgarcia_from_pytorch_to_coreml(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

