import sys
_module = sys.modules[__name__]
del sys
benchmark = _module
check = _module
cpp = _module
jit = _module
lltm = _module
setup = _module
cuda = _module
jit = _module
lltm = _module
setup = _module
grad_check = _module
python = _module
lltm = _module
lltm_baseline = _module

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


import math


import time


import torch


import numpy as np


from torch.utils.cpp_extension import load


from torch import nn


from torch.autograd import Function


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.autograd import gradcheck


import torch.nn.functional as F


def d_elu(z, alpha=1.0):
    e = z.exp()
    mask = alpha * (e - 1) < 0
    return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e)


def d_sigmoid(z):
    s = torch.sigmoid(z)
    return (1 - s) * s


def d_tanh(z):
    t = torch.tanh(z)
    return 1 - t * t


class LLTMFunction(Function):

    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        X = torch.cat([old_h, input], dim=1)
        gate_weights = F.linear(X, weights, bias)
        gates = gate_weights.chunk(3, dim=1)
        input_gate = torch.sigmoid(gates[0])
        output_gate = torch.sigmoid(gates[1])
        candidate_cell = F.elu(gates[2])
        new_cell = old_cell + candidate_cell * input_gate
        new_h = torch.tanh(new_cell) * output_gate
        ctx.save_for_backward(X, weights, input_gate, output_gate, old_cell, new_cell, candidate_cell, gate_weights)
        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        X, weights, input_gate, output_gate, old_cell = ctx.saved_variables[:5]
        new_cell, candidate_cell, gate_weights = ctx.saved_variables[5:]
        d_input = d_weights = d_bias = d_old_h = d_old_cell = None
        d_output_gate = torch.tanh(new_cell) * grad_h
        d_tanh_new_cell = output_gate * grad_h
        d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell
        d_old_cell = d_new_cell
        d_candidate_cell = input_gate * d_new_cell
        d_input_gate = candidate_cell * d_new_cell
        gates = gate_weights.chunk(3, dim=1)
        d_input_gate *= d_sigmoid(gates[0])
        d_output_gate *= d_sigmoid(gates[1])
        d_candidate_cell *= d_elu(gates[2])
        d_gates = torch.cat([d_input_gate, d_output_gate, d_candidate_cell], dim=1)
        if ctx.needs_input_grad[1]:
            d_weights = d_gates.t().mm(X)
        if ctx.needs_input_grad[2]:
            d_bias = d_gates.sum(dim=0, keepdim=True)
        if ctx.needs_input_grad[3] or ctx.needs_input_grad[4]:
            d_X = d_gates.mm(weights)
            state_size = grad_h.shape[1]
            d_old_h, d_input = d_X[:, :state_size], d_X[:, state_size:]
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(nn.Module):

    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = nn.Parameter(torch.Tensor(3 * state_size, input_features + state_size))
        self.bias = nn.Parameter(torch.Tensor(1, 3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)

