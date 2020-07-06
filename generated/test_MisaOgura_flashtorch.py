import sys
_module = sys.modules[__name__]
del sys
flashtorch = _module
activmax = _module
gradient_ascent = _module
saliency = _module
backprop = _module
utils = _module
imagenet = _module
resources = _module
setup = _module
test_backprop = _module
test_gradient_ascent = _module
test_imagenet = _module
test_notebooks = _module
test_utils = _module

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


import torch


import torch.nn as nn


import warnings


import torchvision.transforms as transforms


import torchvision.transforms.functional as F


import inspect


import torch.nn.functional as F


import torchvision.models as models


class CnnGrayscale(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 10, kernel_size=3, stride=3, padding=1)
        self.fc1 = nn.Linear(10 * 25 * 25, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return F.softmax(self.fc1(x.view(-1, 10 * 25 * 25)), dim=1)

