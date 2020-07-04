import sys
_module = sys.modules[__name__]
del sys
models = _module
conf = _module
torchattacks = _module
attack = _module
attacks = _module
apgd = _module
bim = _module
cw = _module
deepfool = _module
fgsm = _module
multiattack = _module
pgd = _module
rfgsm = _module
stepll = _module

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


import torch


import torch.nn as nn


import warnings


import torch.optim as optim


class Holdout(nn.Module):

    def __init__(self):
        super(Holdout, self).__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(3, 32, 5), nn.ReLU(), nn.
            BatchNorm2d(32), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 5), nn.
            BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc_layer = nn.Sequential(nn.Linear(64 * 5 * 5, 100), nn.ReLU(),
            nn.Linear(100, 10))

    def forward(self, x):
        out = self.conv_layer(x)
        out = out.view(-1, 64 * 5 * 5)
        out = self.fc_layer(out)
        return out


class Target(nn.Module):

    def __init__(self):
        super(Target, self).__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(3, 96, 3), nn.GroupNorm(
            32, 96), nn.ELU(), nn.Dropout2d(0.2), nn.Conv2d(96, 96, 3), nn.
            GroupNorm(32, 96), nn.ELU(), nn.Conv2d(96, 96, 3), nn.GroupNorm
            (32, 96), nn.ELU(), nn.Dropout2d(0.5), nn.Conv2d(96, 192, 3),
            nn.GroupNorm(32, 192), nn.ELU(), nn.Conv2d(192, 192, 3), nn.
            GroupNorm(32, 192), nn.ELU(), nn.Dropout2d(0.5), nn.Conv2d(192,
            256, 3), nn.GroupNorm(32, 256), nn.ELU(), nn.Conv2d(256, 256, 1
            ), nn.GroupNorm(32, 256), nn.ELU(), nn.Conv2d(256, 10, 1), nn.
            AvgPool2d(20))

    def forward(self, x):
        out = self.conv_layer(x)
        out = out.view(-1, 10)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Harry24k_adversarial_attacks_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(Target(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

