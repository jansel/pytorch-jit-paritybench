import sys
_module = sys.modules[__name__]
del sys
setup = _module
optimize_lens = _module
propagate = _module
animate_structure = _module
dof = _module
plot_mem = _module
test_grad = _module
models = _module
script_all_params = _module
script_params = _module
vanilla_rnn = _module
vowel_analyze = _module
vowel_helpers = _module
vowel_spectrum = _module
vowel_summary = _module
vowel_train = _module
vowel_train_sklearn = _module
wavetorch = _module
cell = _module
data = _module
vowels = _module
geom = _module
io = _module
operators = _module
plot = _module
probe = _module
rnn = _module
source = _module
train = _module
utils = _module

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


import numpy as np


from torch.nn.functional import conv2d


from torch.nn.functional import pad


import torch.nn as nn


from torch.nn import functional as F


import time


from torch import optim


from torch.utils.data import TensorDataset


from torch.utils.data import DataLoader


from torch.nn.utils.clip_grad import clip_grad_norm_ as clip_grad


from torch.optim.lr_scheduler import StepLR


import math


import random


from torch.nn.utils.rnn import pad_sequence


from copy import deepcopy


from typing import Tuple


class CustomRNN(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, batch_first=
        True, W_scale=0.1, f_hidden=None):
        super(CustomRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.f_hidden = f_hidden
        self.W1 = nn.Parameter((torch.rand(hidden_size, input_size) - 0.5) *
            W_scale)
        self.W2 = nn.Parameter((torch.rand(hidden_size, hidden_size) - 0.5) *
            W_scale)
        self.W3 = nn.Parameter((torch.rand(output_size, hidden_size) - 0.5) *
            W_scale)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        h1 = torch.zeros(x.shape[0], self.hidden_size)
        ys = []
        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):
            h1 = (torch.matmul(self.W2, h1.t()) + torch.matmul(self.W1, xi.t())
                ).t() + self.b_h
            if self.f_hidden is not None:
                h1 = getattr(F, self.f_hidden)(h1)
            y = torch.matmul(self.W3, h1.t()).t()
            ys.append(y)
        ys = torch.stack(ys, dim=1)
        return ys


class CustomRes(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, batch_first=
        True, W_scale=0.1, f_hidden=None):
        super(CustomRes, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.f_hidden = f_hidden
        self.W1 = torch.nn.Parameter((torch.rand(hidden_size, input_size) -
            0.5) * W_scale)
        self.W2 = torch.nn.Parameter((torch.rand(hidden_size, hidden_size) -
            0.5) * W_scale)
        self.W3 = torch.nn.Parameter((torch.rand(output_size, hidden_size) -
            0.5) * W_scale)
        self.b_h = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        h1 = torch.zeros(x.shape[0], self.hidden_size)
        ys = []
        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):
            hprev = h1
            h1 = (torch.matmul(self.W2, h1.t()) + torch.matmul(self.W1, xi.t())
                ).t() + self.b_h
            if self.f_hidden is not None:
                h1 = getattr(F, self.f_hidden)(h1)
            y = torch.matmul(self.W3, h1.t()).t()
            ys.append(y)
            h1 = h1 + hprev
        ys = torch.stack(ys, dim=1)
        return ys


class CustomLSTM(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, batch_first=
        True, W_scale=0.1):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=batch_first)
        self.W3 = torch.nn.Parameter(torch.rand(output_size, hidden_size) - 0.5
            )

    def forward(self, x):
        out, hidden = self.lstm(x.unsqueeze(2))
        ys = torch.matmul(out, self.W3.t())
        return ys


def _laplacian(y, h):
    """Laplacian operator"""
    operator = h ** -2 * torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0],
        [0.0, 1.0, 0.0]]]])
    y = y.unsqueeze(1)
    return conv2d(y, operator, padding=1).squeeze(1)


def _time_step(b, c, y1, y2, dt, h):
    y = torch.mul((dt ** -2 + b * dt ** -1).pow(-1), 2 / dt ** 2 * y1 -
        torch.mul(dt ** -2 - b * dt ** -1, y2) + torch.mul(c.pow(2),
        _laplacian(y1, h)))
    return y


class TimeStep(torch.autograd.Function):

    @staticmethod
    def forward(ctx, b, c, y1, y2, dt, h):
        ctx.save_for_backward(b, c, y1, y2, dt, h)
        return _time_step(b, c, y1, y2, dt, h)

    @staticmethod
    def backward(ctx, grad_output):
        b, c, y1, y2, dt, h = ctx.saved_tensors
        grad_b = grad_c = grad_y1 = grad_y2 = grad_dt = grad_h = None
        if ctx.needs_input_grad[0]:
            grad_b = -(dt * b + 1).pow(-2) * dt * (c.pow(2) * dt ** 2 *
                _laplacian(y1, h) + 2 * y1 - 2 * y2) * grad_output
        if ctx.needs_input_grad[1]:
            grad_c = (b * dt + 1).pow(-1) * (2 * c * dt ** 2 * _laplacian(
                y1, h)) * grad_output
        if ctx.needs_input_grad[2]:
            c2_grad = (b * dt + 1) ** -1 * c.pow(2) * grad_output
            grad_y1 = dt ** 2 * _laplacian(c2_grad, h) + 2 * grad_output * (
                b * dt + 1).pow(-1)
        if ctx.needs_input_grad[3]:
            grad_y2 = (b * dt - 1) * (b * dt + 1).pow(-1) * grad_output
        return grad_b, grad_c, grad_y1, grad_y2, grad_dt, grad_h


def saturable_damping(u, uth, b0):
    return b0 / (1 + torch.abs(u / uth).pow(2))


def to_tensor(x, dtype=None):
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    if type(x) is np.ndarray:
        return torch.from_numpy(x).type(dtype=dtype)
    else:
        return torch.tensor(x, dtype=dtype)


class WaveCell(torch.nn.Module):
    """The recurrent neural network cell implementing the scalar wave equation"""

    def __init__(self, dt: float, geometry, satdamp_b0: float=0.0,
        satdamp_uth: float=0.0, c_nl: float=0.0):
        super().__init__()
        self.register_buffer('dt', to_tensor(dt))
        self.geom = geometry
        self.register_buffer('satdamp_b0', to_tensor(satdamp_b0))
        self.register_buffer('satdamp_uth', to_tensor(satdamp_uth))
        self.register_buffer('c_nl', to_tensor(c_nl))
        cmax = self.geom.cmax
        h = self.geom.h
        if dt > 1 / cmax * h / np.sqrt(2):
            raise ValueError(
                'The spatial discretization defined by the geometry `h = %f` and the temporal discretization defined by the model `dt = %f` do not satisfy the CFL stability criteria'
                 % (h, dt))

    def parameters(self, recursive=True):
        for param in self.geom.parameters():
            yield param

    def forward(self, h1, h2, c_linear, rho):
        """Take a step through time

        Parameters
        ----------
        h1 : 
            Scalar wave field one time step ago (part of the hidden state)
        h2 : 
            Scalar wave field two time steps ago (part of the hidden state)
        c_linear :
            Scalar wave speed distribution (this gets passed in to avoid generating it on each time step, saving memory for backprop)
        rho : 
            Projected density, required for nonlinear response (this gets passed in to avoid generating it on each time step, saving memory for backprop)
        """
        if self.satdamp_b0 > 0:
            b = self.geom.b + rho * saturable_damping(h1, uth=self.
                satdamp_uth, b0=self.satdamp_b0)
        else:
            b = self.geom.b
        if self.c_nl != 0:
            c = c_linear + rho * self.c_nl * h1.pow(2)
        else:
            c = c_linear
        y = TimeStep.apply(b, c, h1, h2, self.dt, self.geom.h)
        return y, h1


class WaveGeometry(torch.nn.Module):

    def __init__(self, domain_shape: Tuple, h: float, c0: float, c1: float,
        abs_N: int=20, abs_sig: float=11, abs_p: float=4.0):
        super().__init__()
        assert len(domain_shape
            ) == 2, 'len(domain_shape) must be equal to 2: only two-dimensional (2D) domains are supported'
        self.domain_shape = domain_shape
        self.register_buffer('h', to_tensor(h))
        self.register_buffer('c0', to_tensor(c0))
        self.register_buffer('c1', to_tensor(c1))
        self.register_buffer('abs_N', to_tensor(abs_N, dtype=torch.uint8))
        self.register_buffer('abs_sig', to_tensor(abs_sig))
        self.register_buffer('abs_p', to_tensor(abs_p, dtype=torch.uint8))
        self._init_b(abs_N, abs_sig, abs_p)

    def state_reconstruction_args(self):
        return {'domain_shape': self.domain_shape, 'h': self.h.item(), 'c0':
            self.c0.item(), 'c1': self.c1.item(), 'abs_N': self.abs_N.item(
            ), 'abs_sig': self.abs_sig.item(), 'abs_p': self.abs_p.item()}

    def __repr__(self):
        return 'WaveGeometry shape={}, h={}'.format(self.domain_shape, self.h)

    def forward(self):
        raise NotImplementedError(
            'WaveGeometry forward() is not implemented. Although WaveGeometry is a subclass of a torch.nn.Module, its forward() method should never be called. It only exists as a torch.nn.Module to hook into pytorch as a component of a WaveCell.'
            )

    @property
    def c(self):
        raise NotImplementedError

    @property
    def b(self):
        return self._b

    @property
    def cmax(self):
        """Helper function for getting the maximum wave speed for calculating CFL"""
        return np.max([self.c0.item(), self.c1.item()])

    def constrain_to_design_region(self):
        pass

    def _init_b(self, abs_N: int, abs_sig: float, abs_p: float):
        """Initialize the distribution of the damping parameter for the PML"""
        Nx, Ny = self.domain_shape
        assert Nx > 2 * abs_N + 1, "The domain isn't large enough in the x-direction to fit absorbing layer. Nx = {} and N = {}".format(
            Nx, abs_N)
        assert Ny > 2 * abs_N + 1, "The domain isn't large enough in the y-direction to fit absorbing layer. Ny = {} and N = {}".format(
            Ny, abs_N)
        b_vals = abs_sig * torch.linspace(0.0, 1.0, abs_N + 1) ** abs_p
        b_x = torch.zeros(Nx, Ny)
        b_y = torch.zeros(Nx, Ny)
        if abs_N > 0:
            b_x[0:abs_N + 1, :] = torch.flip(b_vals, [0]).repeat(Ny, 1
                ).transpose(0, 1)
            b_x[Nx - abs_N - 1:Nx, :] = b_vals.repeat(Ny, 1).transpose(0, 1)
            b_y[:, 0:abs_N + 1] = torch.flip(b_vals, [0]).repeat(Nx, 1)
            b_y[:, Ny - abs_N - 1:Ny] = b_vals.repeat(Nx, 1)
        self.register_buffer('_b', torch.sqrt(b_x ** 2 + b_y ** 2))


class WaveProbe(torch.nn.Module):

    def __init__(self, x, y):
        super().__init__()
        self.register_buffer('x', to_tensor(x, dtype=torch.int64))
        self.register_buffer('y', to_tensor(y, dtype=torch.int64))

    def forward(self, x):
        return x[:, (self.x), (self.y)]

    def plot(self, ax, color='k'):
        marker, = ax.plot(self.x.numpy(), self.y.numpy(), 'o',
            markeredgecolor=color, markerfacecolor='none', markeredgewidth=
            1.0, markersize=4)
        return marker


class WaveRNN(torch.nn.Module):

    def __init__(self, cell, sources, probes=[]):
        super().__init__()
        self.cell = cell
        if type(sources) is list:
            self.sources = torch.nn.ModuleList(sources)
        else:
            self.sources = torch.nn.ModuleList([sources])
        if type(probes) is list:
            self.probes = torch.nn.ModuleList(probes)
        else:
            self.probes = torch.nn.ModuleList([probes])

    def forward(self, x, output_fields=False):
        """Propagate forward in time for the length of the inputs

		Parameters
		----------
		x :
			Input sequence(s), batched in first dimension
		output_fields :
			Override flag for probe output (to get fields)
		"""
        device = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
        batch_size = x.shape[0]
        hidden_state_shape = (batch_size,) + self.cell.geom.domain_shape
        h1 = torch.zeros(hidden_state_shape, device=device)
        h2 = torch.zeros(hidden_state_shape, device=device)
        y_all = []
        c = self.cell.geom.c
        rho = self.cell.geom.rho
        for i, xi in enumerate(x.chunk(x.size(1), dim=1)):
            h1, h2 = self.cell(h1, h2, c, rho)
            for source in self.sources:
                h1 = source(h1, xi.squeeze(-1))
            if len(self.probes) > 0 and not output_fields:
                probe_values = []
                for probe in self.probes:
                    probe_values.append(probe(h1))
                y_all.append(torch.stack(probe_values, dim=-1))
            else:
                y_all.append(h1)
        y = torch.stack(y_all, dim=1)
        return y


class WaveSource(torch.nn.Module):

    def __init__(self, x, y):
        super().__init__()
        self.register_buffer('x', to_tensor(x, dtype=torch.int64))
        self.register_buffer('y', to_tensor(y, dtype=torch.int64))

    def forward(self, Y, X, dt=1.0):
        X_expanded = torch.zeros(Y.size()).detach()
        X_expanded[:, (self.x), (self.y)] = X
        return Y + dt ** 2 * X_expanded

    def plot(self, ax, color='r'):
        marker, = ax.plot(self.x.numpy(), self.y.numpy(), 'o',
            markeredgecolor=color, markerfacecolor='none', markeredgewidth=
            1.0, markersize=4)
        return marker


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_fancompute_wavetorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(WaveProbe(*[], **{'x': 4, 'y': 4}), [torch.rand([4, 5, 5, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(WaveSource(*[], **{'x': 4, 'y': 4}), [torch.rand([4, 5, 5]), torch.rand([4])], {})

