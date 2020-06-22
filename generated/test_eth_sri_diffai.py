import sys
_module = sys.modules[__name__]
del sys
master = _module
__main__ = _module
ai = _module
components = _module
convert = _module
goals = _module
losses = _module
models = _module
scheduling = _module

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


import copy


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.utils.data import Dataset


import torch.onnx


import inspect


from inspect import getargspec


import random


import math


import warnings


from torch.serialization import SourceChangeWarning


import torch.autograd


from functools import reduce


from torch.distributions import multinomial


from torch.distributions import categorical


import abc


from torch.nn.modules.conv import _ConvNd


from enum import Enum


from itertools import count


import numpy as np


class InferModule(nn.Module):

    def __init__(self, *args, normal=False, ibp_init=False, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.infered = False
        self.normal = normal
        self.ibp_init = ibp_init

    def infer(self, in_shape, global_args=None):
        """ this is really actually stateful. """
        if self.infered:
            return self
        self.infered = True
        super(InferModule, self).__init__()
        self.inShape = list(in_shape)
        self.outShape = list(self.init(list(in_shape), *self.args,
            global_args=global_args, **self.kwargs))
        if self.outShape is None:
            raise 'init should set the out_shape'
        self.reset_parameters()
        return self

    def reset_parameters(self):
        if not hasattr(self, 'weight') or self.weight is None:
            return
        n = h.product(self.weight.size()) / self.outShape[0]
        stdv = 1 / math.sqrt(n)
        if self.ibp_init:
            torch.nn.init.orthogonal_(self.weight.data)
        elif self.normal:
            self.weight.data.normal_(0, stdv)
            self.weight.data.clamp_(-1, 1)
        else:
            self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            if self.ibp_init:
                self.bias.data.zero_()
            elif self.normal:
                self.bias.data.normal_(0, stdv)
                self.bias.data.clamp_(-1, 1)
            else:
                self.bias.data.uniform_(-stdv, stdv)

    def clip_norm(self):
        if not hasattr(self, 'weight'):
            return
        if not hasattr(self, 'weight_g'):
            if torch.__version__[0] == '0':
                nn.utils.weight_norm(self, dim=None)
            else:
                nn.utils.weight_norm(self)
        self.weight_g.data.clamp_(-h.max_c_for_norm, h.max_c_for_norm)
        if torch.__version__[0] != '0':
            self.weight_v.data.clamp_(-h.max_c_for_norm * 10000, h.
                max_c_for_norm * 10000)
            if hasattr(self, 'bias'):
                self.bias.data.clamp_(-h.max_c_for_norm * 10000, h.
                    max_c_for_norm * 10000)

    def regularize(self, p):
        reg = 0
        if torch.__version__[0] == '0':
            for param in self.parameters():
                reg += param.norm(p)
        else:
            if hasattr(self, 'weight_g'):
                reg += self.weight_g.norm().sum()
                reg += self.weight_v.norm().sum()
            elif hasattr(self, 'weight'):
                reg += self.weight.norm().sum()
            if hasattr(self, 'bias'):
                reg += self.bias.view(-1).norm(p=p).sum()
        return reg

    def remove_norm(self):
        if hasattr(self, 'weight_g'):
            torch.nn.utils.remove_weight_norm(self)

    def showNet(self, t=''):
        None

    def printNet(self, f):
        None

    @abc.abstractmethod
    def forward(self, *args, **kargs):
        pass

    def __call__(self, *args, onyx=False, **kargs):
        if onyx:
            return self.forward(*args, onyx=onyx, **kargs)
        else:
            return super(InferModule, self).__call__(*args, **kargs)

    @abc.abstractmethod
    def neuronCount(self):
        pass

    def depth(self):
        return 0


class AbstractNet(nn.Module):

    def __init__(self, domain, net, abstractNet):
        super(AbstractNet, self).__init__()
        self.net = net
        self.abstractNet = abstractNet
        if hasattr(domain, 'net') and domain.net is not None:
            self.netDom = domain.net

    def forward(self, inpt):
        return self.abstractNet(inpt)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_eth_sri_diffai(_paritybench_base):
    pass
    def test_000(self):
        self._check(AbstractNet(*[], **{'domain': 4, 'net': 4, 'abstractNet': ReLU()}), [torch.rand([4, 4, 4, 4])], {})

