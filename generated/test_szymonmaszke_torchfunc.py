import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
tests = _module
cuda_test = _module
recorders_test = _module
registrators_test = _module
module_test = _module
performance = _module
layers_test = _module
performance_test = _module
technology_test = _module
record_test = _module
torchfunc_test = _module
torchfunc = _module
_base = _module
_dev_utils = _module
_general = _module
cuda = _module
hooks = _module
_dev_utils = _module
recorders = _module
registrators = _module
module = _module
performance = _module
layers = _module
technology = _module

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


import torch


import itertools


import time


import typing


from functools import wraps


import numpy as np


import copy


import abc


import collections


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.convolution = torch.nn.Sequential(torch.nn.Conv2d(1, 32, 3), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(32, 128, 3, groups=32), torch.nn.ReLU(inplace=True), torch.nn.Conv2d(128, 250, 3))
        self.classifier = torch.nn.Sequential(torch.nn.Linear(250, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10))

    def forward(self, inputs):
        convolved = torch.nn.AdaptiveAvgPool2d(1)(self.convolution(inputs)).flatten()
        return self.classifier(convolved)

