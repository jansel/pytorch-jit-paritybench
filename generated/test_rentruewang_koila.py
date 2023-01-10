import sys
_module = sys.modules[__name__]
del sys
examples = _module
koila = _module
constants = _module
delayed = _module
core = _module
delayed = _module
eager = _module
errors = _module
gpus = _module
interfaces = _module
components = _module
arithmetic = _module
datatypes = _module
indexible = _module
meminfo = _module
multidim = _module
withbatch = _module
runnable = _module
tensorlike = _module
prepasses = _module
shapes = _module
tensors = _module
tests = _module
test_layers = _module
test_lazy = _module
test_models = _module
test_prepasses = _module
utils = _module

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


import logging


import torch


from torch import Tensor


from torch.nn import CrossEntropyLoss


from torch.nn import Flatten


from torch.nn import Linear


from torch.nn import Module


from torch.nn import ReLU


from torch.nn import Sequential


from typing import Dict


from torch import dtype


import functools


from functools import wraps


from typing import Any


from typing import Callable


from typing import Generic


from typing import List


from typing import NamedTuple


from typing import NoReturn


from typing import Sequence


from typing import Tuple


from typing import Type


from typing import TypeVar


from typing import final


from typing import overload


from torch import cuda


from torch import device as Device


from torch import dtype as DType


from numpy import ndarray


import math


from typing import Generator


from abc import abstractmethod


from typing import Protocol


from typing import Union


from typing import runtime_checkable


from typing import Literal


from torch.nn import AvgPool1d


from torch.nn import AvgPool2d


from torch.nn import AvgPool3d


from torch.nn import BatchNorm1d


from torch.nn import BatchNorm2d


from torch.nn import BatchNorm3d


from torch.nn import Conv1d


from torch.nn import Conv2d


from torch.nn import Conv3d


from torch.nn import ConvTranspose1d


from torch.nn import ConvTranspose2d


from torch.nn import ConvTranspose3d


from torch.nn import Dropout


from torch.nn import Embedding


from torch.nn import LayerNorm


from torch.nn import LeakyReLU


from torch.nn import MaxPool1d


from torch.nn import MaxPool2d


from torch.nn import MaxPool3d


from torch.nn import Sigmoid


from torch.nn import Softmax


import typing


import numpy as np


from torch.nn import functional as F


from torch.types import Number


class NeuralNetwork(Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = Flatten()
        self.linear_relu_stack = Sequential(Linear(28 * 28, 512), ReLU(), Linear(512, 512), ReLU(), Linear(512, 10))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

