import sys
_module = sys.modules[__name__]
del sys
controldiffeq = _module
cdeint_module = _module
interpolate = _module
misc = _module
example = _module
common = _module
datasets = _module
common = _module
sepsis = _module
speech_commands = _module
uea = _module
models = _module
metamodel = _module
other = _module
vector_fields = _module
parse_results = _module
sepsis = _module
setup = _module

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


import torch


import math


import numpy as np


import copy


import sklearn.metrics


import sklearn.model_selection


import torchaudio


import collections as co


import matplotlib.pyplot as plt


class VectorField(torch.nn.Module):

    def __init__(self, dX_dt, func):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func: As cdeint.
        """
        super(VectorField, self).__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError('func must be a torch.nn.Module.')
        self.dX_dt = dX_dt
        self.func = func

    def __call__(self, t, z):
        control_gradient = self.dX_dt(t)
        vector_field = self.func(z)
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        return out


class CDEFunc(torch.nn.Module):

    def __init__(self, input_channels, hidden_channels):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    def forward(self, z):
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        z = z.tanh()
        z = z.view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        return z


class ContinuousRNNConverter(torch.nn.Module):

    def __init__(self, input_channels, hidden_channels, model):
        super(ContinuousRNNConverter, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.model = model
        out_base = torch.zeros(self.input_channels + self.hidden_channels, self.input_channels)
        for i in range(self.input_channels):
            out_base[i, i] = 1
        self.register_buffer('out_base', out_base)

    def extra_repr(self):
        return 'input_channels: {}, hidden_channels: {}'.format(self.input_channels, self.hidden_channels)

    def forward(self, z):
        x = z[..., :self.input_channels]
        h = z[..., self.input_channels:]
        h = h.clamp(-1, 1)
        model_out = self.model(x, h)
        batch_dims = model_out.shape[:-1]
        out = self.out_base.repeat(*batch_dims, 1, 1).clone()
        out[..., self.input_channels:, 0] = model_out
        return out


class NeuralCDE(torch.nn.Module):
    """A Neural CDE model. Provides a wrapper around the lower-level cdeint function, to get a flexible Neural CDE
    model.

    Specifically, considering the CDE
    ```
    z_t = z_{t_0} + \\int_{t_0}^t f(z_s)dX_s
    ```
    where X is determined by the data, and given some terminal time t_N, then this model first computes z_{t_N}, then
    performs a linear function on it, and then outputs the result.

    It's known that linear functions on CDEs are universal approximators, so this is a very general type of model.
    """

    def __init__(self, func, input_channels, hidden_channels, output_channels, initial=True):
        """
        Arguments:
            func: As cdeint.
            input_channels: How many channels there are in the input.
            hidden_channels: The number of hidden channels, i.e. the size of z_t.
            output_channels: How many channels to perform a linear map to at the end.
            initial: Whether to automatically construct the initial value from data (in which case z0 must not be passed
                during forward()), or to use the one supplied during forward (in which case z0 must be passed during
                forward()).
        """
        if isinstance(func, ContinuousRNNConverter):
            hidden_channels = hidden_channels + input_channels
        super(NeuralCDE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.func = func
        self.initial = initial
        if initial and not isinstance(func, ContinuousRNNConverter):
            self.initial_network = torch.nn.Linear(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

    def extra_repr(self):
        return 'input_channels={}, hidden_channels={}, output_channels={}, initial={}'.format(self.input_channels, self.hidden_channels, self.output_channels, self.initial)

    def forward(self, times, coeffs, final_index, z0=None, stream=False, **kwargs):
        """
        Arguments:
            times: The times of the observations for the input path X, e.g. as passed as an argument to
                `controldiffeq.natural_cubic_spline_coeffs`.
            coeffs: The coefficients describing the input path X, e.g. as returned by
                `controldiffeq.natural_cubic_spline_coeffs`.
            final_index: Each batch element may have a different final time. This defines the index within the tensor
                `times` of where the final time for each batch element is.
            z0: See the 'initial' argument to __init__.
            stream: Whether to return the result of the Neural CDE model at all times (True), or just the final time
                (False). Defaults to just the final time. The `final_index` argument is ignored if stream is True.
            **kwargs: Will be passed to cdeint.

        Returns:
            If stream is False, then this will return the terminal time z_T. If stream is True, then this will return
            all intermediate times z_t, for those t for which there was data.
        """
        coeff, _, _, _ = coeffs
        batch_dims = coeff.shape[:-2]
        if not stream:
            assert batch_dims == final_index.shape, 'coeff.shape[:-2] must be the same as final_index.shape. coeff.shape[:-2]={}, final_index.shape={}'.format(batch_dims, final_index.shape)
        cubic_spline = controldiffeq.NaturalCubicSpline(times, coeffs)
        if z0 is None:
            assert self.initial, 'Was not expecting to be given no value of z0.'
            if isinstance(self.func, ContinuousRNNConverter):
                z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=coeff.dtype, device=coeff.device)
            else:
                z0 = self.initial_network(cubic_spline.evaluate(times[0]))
        else:
            assert not self.initial, 'Was expecting to be given a value of z0.'
            if isinstance(self.func, ContinuousRNNConverter):
                z0_extra = torch.zeros(*batch_dims, self.input_channels, dtype=z0.dtype, device=z0.device)
                z0 = torch.cat([z0_extra, z0], dim=-1)
        if stream:
            t = times
        else:
            sorted_final_index, inverse_final_index = final_index.unique(sorted=True, return_inverse=True)
            if 0 in sorted_final_index:
                sorted_final_index = sorted_final_index[1:]
                final_index = inverse_final_index
            else:
                final_index = inverse_final_index + 1
            if len(times) - 1 in sorted_final_index:
                sorted_final_index = sorted_final_index[:-1]
            t = torch.cat([times[0].unsqueeze(0), times[sorted_final_index], times[-1].unsqueeze(0)])
        if 'method' not in kwargs:
            kwargs['method'] = 'rk4'
        if kwargs['method'] == 'rk4':
            if 'options' not in kwargs:
                kwargs['options'] = {}
            options = kwargs['options']
            if 'step_size' not in options and 'grid_constructor' not in options:
                time_diffs = times[1:] - times[:-1]
                options['step_size'] = time_diffs.min().item()
        z_t = controldiffeq.cdeint(dX_dt=cubic_spline.derivative, z0=z0, func=self.func, t=t, **kwargs)
        if stream:
            for i in range(len(z_t.shape) - 2, 0, -1):
                z_t = z_t.transpose(0, i)
        else:
            final_index_indices = final_index.unsqueeze(-1).expand(z_t.shape[1:]).unsqueeze(0)
            z_t = z_t.gather(dim=0, index=final_index_indices).squeeze(0)
        pred_y = self.linear(z_t)
        return pred_y


class _SqueezeEnd(torch.nn.Module):

    def __init__(self, model):
        super(_SqueezeEnd, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs).squeeze(-1)


class _GRU(torch.nn.Module):

    def __init__(self, input_channels, hidden_channels, output_channels, use_intensity):
        super(_GRU, self).__init__()
        assert input_channels % 2 == 1, 'Input channels must be odd: 1 for time, plus 1 for each actual input, plus 1 for whether an observation was made for the actual input.'
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.use_intensity = use_intensity
        gru_channels = input_channels if use_intensity else (input_channels - 1) // 2
        self.gru_cell = torch.nn.GRUCell(input_size=gru_channels, hidden_size=hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, output_channels)

    def extra_repr(self):
        return 'input_channels={}, hidden_channels={}, output_channels={}, use_intensity={}'.format(self.input_channels, self.hidden_channels, self.output_channels, self.use_intensity)

    def evolve(self, h, time_diff):
        raise NotImplementedError

    def _step(self, Xi, h, dt, half_num_channels):
        observation = Xi[:, 1:1 + half_num_channels].max(dim=1).values > 0.5
        if observation.any():
            Xi_piece = Xi if self.use_intensity else Xi[:, 1 + half_num_channels:]
            Xi_piece = Xi_piece.clone()
            Xi_piece[:, 0] += dt
            new_h = self.gru_cell(Xi_piece, h)
            h = torch.where(observation.unsqueeze(1), new_h, h)
            dt += torch.where(observation, torch.tensor(0.0, dtype=Xi.dtype, device=Xi.device), Xi[:, 0])
        return h, dt

    def forward(self, times, coeffs, final_index, z0=None):
        interp = controldiffeq.NaturalCubicSpline(times, coeffs)
        X = torch.stack([interp.evaluate(t) for t in times], dim=-2)
        half_num_channels = (self.input_channels - 1) // 2
        X[:, 1:, 1:1 + half_num_channels] -= X[:, :-1, 1:1 + half_num_channels]
        X[:, 0, 0] -= times[0]
        X[:, 1:, 0] -= times[:-1]
        batch_dims = X.shape[:-2]
        if z0 is None:
            z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=X.dtype, device=X.device)
        X_unbound = X.unbind(dim=1)
        h, dt = self._step(X_unbound[0], z0, torch.zeros(*batch_dims, dtype=X.dtype, device=X.device), half_num_channels)
        hs = [h]
        time_diffs = times[1:] - times[:-1]
        for time_diff, Xi in zip(time_diffs, X_unbound[1:]):
            h = self.evolve(h, time_diff)
            h, dt = self._step(Xi, h, dt, half_num_channels)
            hs.append(h)
        out = torch.stack(hs, dim=1)
        final_index_indices = final_index.unsqueeze(-1).expand(out.size(0), out.size(2)).unsqueeze(1)
        final_out = out.gather(dim=1, index=final_index_indices).squeeze(1)
        return self.linear(final_out)


class GRU_dt(_GRU):

    def evolve(self, h, time_diff):
        return h


class GRU_D(_GRU):

    def __init__(self, input_channels, hidden_channels, output_channels, use_intensity):
        super(GRU_D, self).__init__(input_channels=input_channels, hidden_channels=hidden_channels, output_channels=output_channels, use_intensity=use_intensity)
        self.decay = torch.nn.Linear(1, hidden_channels)

    def evolve(self, h, time_diff):
        return h * torch.exp(-self.decay(time_diff.unsqueeze(0)).squeeze(0).relu())


class _ODERNNFunc(torch.nn.Module):

    def __init__(self, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(_ODERNNFunc, self).__init__()
        layers = [torch.nn.Linear(hidden_channels, hidden_hidden_channels)]
        for _ in range(num_hidden_layers - 1):
            layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels))
        layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(hidden_hidden_channels, hidden_channels))
        self.sequential = torch.nn.Sequential(*layers)

    def forward(self, t, x):
        return self.sequential(x)


class ODERNN(_GRU):

    def __init__(self, input_channels, hidden_channels, output_channels, hidden_hidden_channels, num_hidden_layers, use_intensity):
        super(ODERNN, self).__init__(input_channels=input_channels, hidden_channels=hidden_channels, output_channels=output_channels, use_intensity=use_intensity)
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.func = _ODERNNFunc(hidden_channels, hidden_hidden_channels, num_hidden_layers)

    def extra_repr(self):
        return 'hidden_hidden_channels={}, num_hidden_layers={}'.format(self.hidden_hidden_channels, self.num_hidden_layers)

    def evolve(self, h, time_diff):
        t = torch.tensor([0, time_diff.item()], dtype=time_diff.dtype, device=time_diff.device)
        out = torchdiffeq.odeint_adjoint(func=self.func, y0=h, t=t, method='rk4')
        return out[1]


class SingleHiddenLayer(torch.nn.Module):

    def __init__(self, input_channels, hidden_channels):
        super(SingleHiddenLayer, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    def extra_repr(self):
        return 'input_channels: {}, hidden_channels: {}'.format(self.input_channels, self.hidden_channels)

    def forward(self, z):
        z = self.linear1(z)
        z = torch.relu(z)
        z = self.linear2(z)
        z = z.view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        return z


class FinalTanh(torch.nn.Module):

    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = torch.nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels) for _ in range(num_hidden_layers - 1))
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels)

    def extra_repr(self):
        return 'input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}'.format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        z = z.tanh()
        return z


class _GRU_ODE(torch.nn.Module):

    def __init__(self, input_channels, hidden_channels):
        super(_GRU_ODE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.W_r = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.W_z = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.W_h = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.U_r = torch.nn.Linear(hidden_channels, hidden_channels)
        self.U_z = torch.nn.Linear(hidden_channels, hidden_channels)
        self.U_h = torch.nn.Linear(hidden_channels, hidden_channels)

    def extra_repr(self):
        return 'input_channels: {}, hidden_channels: {}'.format(self.input_channels, self.hidden_channels)

    def forward(self, x, h):
        r = self.W_r(x) + self.U_r(h)
        r = r.sigmoid()
        z = self.W_z(x) + self.U_z(h)
        z = z.sigmoid()
        g = self.W_h(x) + self.U_h(r * h)
        g = g.tanh()
        return (1 - z) * (g - h)


class InitialValueNetwork(torch.nn.Module):

    def __init__(self, intensity, hidden_channels, model):
        super(InitialValueNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(7 if intensity else 5, 256)
        self.linear2 = torch.nn.Linear(256, hidden_channels)
        self.model = model

    def forward(self, times, coeffs, final_index, **kwargs):
        *coeffs, static = coeffs
        z0 = self.linear1(static)
        z0 = z0.relu()
        z0 = self.linear2(z0)
        return self.model(times, coeffs, final_index, z0=z0, **kwargs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CDEFunc,
     lambda: ([], {'input_channels': 4, 'hidden_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FinalTanh,
     lambda: ([], {'input_channels': 4, 'hidden_channels': 4, 'hidden_hidden_channels': 4, 'num_hidden_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SingleHiddenLayer,
     lambda: ([], {'input_channels': 4, 'hidden_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_GRU_ODE,
     lambda: ([], {'input_channels': 4, 'hidden_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (_ODERNNFunc,
     lambda: ([], {'hidden_channels': 4, 'hidden_hidden_channels': 4, 'num_hidden_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (_SqueezeEnd,
     lambda: ([], {'model': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
]

class Test_patrick_kidger_NeuralCDE(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

