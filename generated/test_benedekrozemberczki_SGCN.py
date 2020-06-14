import sys
_module = sys.modules[__name__]
del sys
main = _module
param_parser = _module
sgcn = _module
signedsageconvolution = _module
utils = _module

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


import time


import torch


import random


import numpy as np


import torch.nn.init as init


from torch.nn import Parameter


import torch.nn.functional as F


import math


class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """

    def __init__(self, *args):
        """
        Model initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)


def uniform(size, tensor):
    """
    Uniform weight initialization.
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


class SignedSAGEConvolution(torch.nn.Module):
    """
    Abstract Signed SAGE convolution class.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param norm_embed: Normalize embedding -- boolean.
    :param bias: Add bias or no.
    """

    def __init__(self, in_channels, out_channels, norm=True, norm_embed=
        True, bias=True):
        super(SignedSAGEConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.norm_embed = norm_embed
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        size = self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def __repr__(self):
        """
        Create formal string representation.
        """
        return '{}({}, {})'.format(self.__class__.__name__, self.
            in_channels, self.out_channels)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_benedekrozemberczki_SGCN(_paritybench_base):
    pass
