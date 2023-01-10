import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
feed_forward = _module
palette_fit = _module
palette_swap = _module
show_cmaps = _module
zennit = _module
attribution = _module
canonizers = _module
cmap = _module
composites = _module
core = _module
image = _module
layer = _module
rules = _module
torchvision = _module
types = _module
conftest = _module
helpers = _module
test_attribution = _module
test_canonizers = _module
test_cmap = _module
test_composites = _module
test_core = _module
test_image = _module
test_rules = _module
test_torchvision = _module

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


import re


from functools import partial


import torch


import numpy as np


from torch.utils.data import DataLoader


from torch.utils.data import Subset


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from torchvision.transforms import CenterCrop


from torchvision.transforms import ToTensor


from torchvision.datasets import ImageFolder


from torchvision.models import vgg16


from torchvision.models import vgg16_bn


from torchvision.models import resnet50


from abc import ABCMeta


from abc import abstractmethod


from itertools import product


import functools


from torchvision.models.resnet import Bottleneck as ResNetBottleneck


from torchvision.models.resnet import BasicBlock as ResNetBasicBlock


from collections import OrderedDict


from torch.nn import Conv1d


from torch.nn import ConvTranspose1d


from torch.nn import Linear


from torch.nn import Conv2d


from torch.nn import ConvTranspose2d


from torch.nn import Conv3d


from torch.nn import ConvTranspose3d


from torch.nn import BatchNorm1d


from torch.nn import BatchNorm2d


from torch.nn import BatchNorm3d


from torchvision.models import vgg11


from torchvision.models import resnet18


from torchvision.models import alexnet


from torch.nn import Sequential


from itertools import islice


from functools import wraps


from copy import deepcopy


from torchvision.models import vgg11_bn


class Sum(torch.nn.Module):
    """Compute the sum along an axis.

    Parameters
    ----------
    dim : int
        Dimension over which to sum.
    """

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        """Computes the sum along a dimension."""
        return torch.sum(input, dim=self.dim)


class IdentityLogger(torch.nn.Module):
    """Helper-Module to log input tensors."""

    def __init__(self):
        super().__init__()
        self.tensors = []

    def forward(self, input):
        """Clone input, append to self.tensors and return the cloned tensor."""
        self.tensors.append(input.clone())
        return self.tensors[-1]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (IdentityLogger,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Sum,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_chr5tphr_zennit(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

