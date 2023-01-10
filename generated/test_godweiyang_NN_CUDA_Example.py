import sys
_module = sys.modules[__name__]
del sys
setup = _module
time = _module
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


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import time


import numpy as np


import torch


from torch import nn


from torch.autograd import Function


class AddModelFunction(Function):

    @staticmethod
    def forward(ctx, a, b, n):
        c = torch.empty(n)
        if args.compiler == 'jit':
            cuda_module.torch_launch_add2(c, a, b, n)
        elif args.compiler == 'setup':
            add2.torch_launch_add2(c, a, b, n)
        elif args.compiler == 'cmake':
            torch.ops.add2.torch_launch_add2(c, a, b, n)
        else:
            raise Exception('Type of cuda compiler must be one of jit/setup/cmake.')
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output, None


class AddModel(nn.Module):

    def __init__(self, n):
        super(AddModel, self).__init__()
        self.n = n
        self.a = nn.Parameter(torch.Tensor(self.n))
        self.b = nn.Parameter(torch.Tensor(self.n))
        self.a.data.normal_(mean=0.0, std=1.0)
        self.b.data.normal_(mean=0.0, std=1.0)

    def forward(self):
        a2 = torch.square(self.a)
        b2 = torch.square(self.b)
        c = AddModelFunction.apply(a2, b2, self.n)
        return c

