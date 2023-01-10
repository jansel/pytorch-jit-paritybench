import sys
_module = sys.modules[__name__]
del sys
conf = _module
atari_ppo = _module
atari_tf = _module
atari_torch = _module
bidirectional = _module
infer_states = _module
keras_save = _module
pd_example = _module
pt_example = _module
pt_implicit = _module
save_model = _module
stacking_ncp = _module
torch_cfc_sinusoidal = _module
ncps = _module
datasets = _module
icra2020_lidar_collision_avoidance = _module
tf = _module
atari_cloning = _module
atari_cloning = _module
utils = _module
paddle = _module
ltc_cell = _module
test_tf = _module
test_torch = _module
cfc = _module
cfc_cell = _module
ltc = _module
mm_rnn = _module
wired_cfc_cell = _module
cfc = _module
cfc_cell = _module
lstm = _module
ltc = _module
ltc_cell = _module
wired_cfc_cell = _module
wirings = _module
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


from torch.utils.data import Dataset


import torch.optim as optim


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


import torch.utils.data as data


import time


from torch import nn


from typing import Optional


from typing import Union


class ConvBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 64, 5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
        self.conv3 = nn.Conv2d(128, 128, 5, padding=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 5, padding=2, stride=2)
        self.norm = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.mean((-1, -2))
        x = self.norm(x)
        return x


class LeCun(nn.Module):

    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class CfCCell(nn.Module):

    def __init__(self, input_size, hidden_size, mode='default', backbone_activation='lecun_tanh', backbone_units=128, backbone_layers=1, backbone_dropout=0.0, sparsity_mask=None):
        """A `Closed-form Continuous-time <https://arxiv.org/abs/2106.13898>`_ cell.

        .. Note::
            This is an RNNCell that process single time-steps. To get a full RNN that can process sequences see `ncps.torch.CfC`.



        :param input_size:
        :param hidden_size:
        :param mode:
        :param backbone_activation:
        :param backbone_units:
        :param backbone_layers:
        :param backbone_dropout:
        :param sparsity_mask:
        """
        super(CfCCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        allowed_modes = ['default', 'pure', 'no_gate']
        if mode not in allowed_modes:
            raise ValueError(f"Unknown mode '{mode}', valid options are {str(allowed_modes)}")
        self.sparsity_mask = None if sparsity_mask is None else torch.nn.Parameter(data=torch.from_numpy(np.abs(sparsity_mask.T).astype(np.float32)), requires_grad=False)
        self.mode = mode
        if backbone_activation == 'silu':
            backbone_activation = nn.SiLU
        elif backbone_activation == 'relu':
            backbone_activation = nn.ReLU
        elif backbone_activation == 'tanh':
            backbone_activation = nn.Tanh
        elif backbone_activation == 'gelu':
            backbone_activation = nn.GELU
        elif backbone_activation == 'lecun_tanh':
            backbone_activation = LeCun
        else:
            raise ValueError(f'Unknown activation {backbone_activation}')
        self.backbone = None
        self.backbone_layers = backbone_layers
        if backbone_layers > 0:
            layer_list = [nn.Linear(input_size + hidden_size, backbone_units), backbone_activation()]
            for i in range(1, backbone_layers):
                layer_list.append(nn.Linear(backbone_units, backbone_units))
                layer_list.append(backbone_activation())
                if backbone_dropout > 0.0:
                    layer_list.append(torch.nn.Dropout(backbone_dropout))
            self.backbone = nn.Sequential(*layer_list)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        cat_shape = int(self.hidden_size + input_size if backbone_layers == 0 else backbone_units)
        self.ff1 = nn.Linear(cat_shape, hidden_size)
        if self.mode == 'pure':
            self.w_tau = torch.nn.Parameter(data=torch.zeros(1, self.hidden_size), requires_grad=True)
            self.A = torch.nn.Parameter(data=torch.ones(1, self.hidden_size), requires_grad=True)
        else:
            self.ff2 = nn.Linear(cat_shape, hidden_size)
            self.time_a = nn.Linear(cat_shape, hidden_size)
            self.time_b = nn.Linear(cat_shape, hidden_size)
        self.init_weights()

    def init_weights(self):
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                torch.nn.init.xavier_uniform_(w)

    def forward(self, input, hx, ts):
        x = torch.cat([input, hx], 1)
        if self.backbone_layers > 0:
            x = self.backbone(x)
        if self.sparsity_mask is not None:
            ff1 = F.linear(x, self.ff1.weight * self.sparsity_mask, self.ff1.bias)
        else:
            ff1 = self.ff1(x)
        if self.mode == 'pure':
            new_hidden = -self.A * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1))) * ff1 + self.A
        else:
            if self.sparsity_mask is not None:
                ff2 = F.linear(x, self.ff2.weight * self.sparsity_mask, self.ff2.bias)
            else:
                ff2 = self.ff2(x)
            ff1 = self.tanh(ff1)
            ff2 = self.tanh(ff2)
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = self.sigmoid(t_a * ts + t_b)
            if self.mode == 'no_gate':
                new_hidden = ff1 + t_interp * ff2
            else:
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
        return new_hidden, new_hidden


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_map = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.recurrent_map = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for w in self.input_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.xavier_uniform_(w)
        for w in self.recurrent_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.orthogonal_(w)

    def forward(self, inputs, states):
        output_state, cell_state = states
        z = self.input_map(inputs) + self.recurrent_map(output_state)
        i, ig, fg, og = z.chunk(4, 1)
        input_activation = self.tanh(i)
        input_gate = self.sigmoid(ig)
        forget_gate = self.sigmoid(fg + 1.0)
        output_gate = self.sigmoid(og)
        new_cell = cell_state * forget_gate + input_activation * input_gate
        output_state = self.tanh(new_cell) * output_gate
        return output_state, new_cell


class WiredCfCCell(nn.Module):

    def __init__(self, input_size, wiring, mode='default'):
        super(WiredCfCCell, self).__init__()
        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError("Wiring error! Unknown number of input features. Please pass the parameter 'input_size' or call the 'wiring.build()'.")
        self._wiring = wiring
        self._layers = []
        in_features = wiring.input_dim
        for l in range(wiring.num_layers):
            hidden_units = self._wiring.get_neurons_of_layer(l)
            if l == 0:
                input_sparsity = self._wiring.sensory_adjacency_matrix[:, hidden_units]
            else:
                prev_layer_neurons = self._wiring.get_neurons_of_layer(l - 1)
                input_sparsity = self._wiring.adjacency_matrix[:, hidden_units]
                input_sparsity = input_sparsity[prev_layer_neurons, :]
            input_sparsity = np.concatenate([input_sparsity, np.ones((len(hidden_units), len(hidden_units)))], axis=0)
            rnn_cell = CfCCell(in_features, len(hidden_units), mode, backbone_activation='lecun_tanh', backbone_units=0, backbone_layers=0, backbone_dropout=0.0, sparsity_mask=input_sparsity)
            self.register_module(f'layer_{l}', rnn_cell)
            self._layers.append(rnn_cell)
            in_features = len(hidden_units)

    @property
    def state_size(self):
        return self._wiring.units

    @property
    def layer_sizes(self):
        return [len(self._wiring.get_neurons_of_layer(i)) for i in range(self._wiring.num_layers)]

    @property
    def num_layers(self):
        return self._wiring.num_layers

    @property
    def sensory_size(self):
        return self._wiring.input_dim

    @property
    def motor_size(self):
        return self._wiring.output_dim

    @property
    def output_size(self):
        return self.motor_size

    @property
    def synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    def forward(self, input, hx, timespans):
        h_state = torch.split(hx, self.layer_sizes, dim=1)
        new_h_state = []
        inputs = input
        for i in range(self.num_layers):
            h, _ = self._layers[i].forward(inputs, h_state[i], timespans)
            inputs = h
            new_h_state.append(h)
        new_h_state = torch.cat(new_h_state, dim=1)
        return h, new_h_state


class ConvCfC(nn.Module):

    def __init__(self, n_actions):
        super().__init__()
        self.conv_block = ConvBlock()
        self.rnn = CfC(256, 64, batch_first=True, proj_size=n_actions)

    def forward(self, x, hx=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        x = self.conv_block(x)
        x = x.view(batch_size, seq_len, *x.shape[1:])
        x, hx = self.rnn(x, hx)
        return x, hx


class RNNSequence(nn.Module):

    def __init__(self, rnn_cell):
        super(RNNSequence, self).__init__()
        self.rnn_cell = rnn_cell

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden_state = torch.zeros((batch_size, self.rnn_cell.state_size), device=device)
        outputs = []
        for t in range(seq_len):
            inputs = x[:, t]
            new_output, hidden_state = self.rnn_cell.forward(inputs, hidden_state)
            outputs.append(new_output)
        outputs = torch.stack(outputs, dim=1)
        return outputs


class LTCCell(nn.Module):

    def __init__(self, wiring, in_features=None, input_mapping='affine', output_mapping='affine', ode_unfolds=6, epsilon=1e-08, implicit_param_constraints=False):
        """A `Liquid time-constant (LTC) <https://ojs.aaai.org/index.php/AAAI/article/view/16936>`_ cell.

        .. Note::
            This is an RNNCell that process single time-steps. To get a full RNN that can process sequences see `ncps.torch.LTC`.


        :param wiring:
        :param in_features:
        :param input_mapping:
        :param output_mapping:
        :param ode_unfolds:
        :param epsilon:
        :param implicit_param_constraints:
        """
        super(LTCCell, self).__init__()
        if in_features is not None:
            wiring.build(in_features)
        if not wiring.is_built():
            raise ValueError("Wiring error! Unknown number of input features. Please pass the parameter 'in_features' or call the 'wiring.build()'.")
        self.make_positive_fn = nn.Softplus() if implicit_param_constraints else nn.Identity()
        self._implicit_param_constraints = implicit_param_constraints
        self._init_ranges = {'gleak': (0.001, 1.0), 'vleak': (-0.2, 0.2), 'cm': (0.4, 0.6), 'w': (0.001, 1.0), 'sigma': (3, 8), 'mu': (0.3, 0.8), 'sensory_w': (0.001, 1.0), 'sensory_sigma': (3, 8), 'sensory_mu': (0.3, 0.8)}
        self._wiring = wiring
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self._clip = torch.nn.ReLU()
        self._allocate_parameters()

    @property
    def state_size(self):
        return self._wiring.units

    @property
    def sensory_size(self):
        return self._wiring.input_dim

    @property
    def motor_size(self):
        return self._wiring.output_dim

    @property
    def output_size(self):
        return self.motor_size

    @property
    def synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    def add_weight(self, name, init_value, requires_grad=True):
        param = torch.nn.Parameter(init_value, requires_grad=requires_grad)
        self.register_parameter(name, param)
        return param

    def _get_init_value(self, shape, param_name):
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval

    def _allocate_parameters(self):
        None
        self._params = {}
        self._params['gleak'] = self.add_weight(name='gleak', init_value=self._get_init_value((self.state_size,), 'gleak'))
        self._params['vleak'] = self.add_weight(name='vleak', init_value=self._get_init_value((self.state_size,), 'vleak'))
        self._params['cm'] = self.add_weight(name='cm', init_value=self._get_init_value((self.state_size,), 'cm'))
        self._params['sigma'] = self.add_weight(name='sigma', init_value=self._get_init_value((self.state_size, self.state_size), 'sigma'))
        self._params['mu'] = self.add_weight(name='mu', init_value=self._get_init_value((self.state_size, self.state_size), 'mu'))
        self._params['w'] = self.add_weight(name='w', init_value=self._get_init_value((self.state_size, self.state_size), 'w'))
        self._params['erev'] = self.add_weight(name='erev', init_value=torch.Tensor(self._wiring.erev_initializer()))
        self._params['sensory_sigma'] = self.add_weight(name='sensory_sigma', init_value=self._get_init_value((self.sensory_size, self.state_size), 'sensory_sigma'))
        self._params['sensory_mu'] = self.add_weight(name='sensory_mu', init_value=self._get_init_value((self.sensory_size, self.state_size), 'sensory_mu'))
        self._params['sensory_w'] = self.add_weight(name='sensory_w', init_value=self._get_init_value((self.sensory_size, self.state_size), 'sensory_w'))
        self._params['sensory_erev'] = self.add_weight(name='sensory_erev', init_value=torch.Tensor(self._wiring.sensory_erev_initializer()))
        self._params['sparsity_mask'] = self.add_weight('sparsity_mask', torch.Tensor(np.abs(self._wiring.adjacency_matrix)), requires_grad=False)
        self._params['sensory_sparsity_mask'] = self.add_weight('sensory_sparsity_mask', torch.Tensor(np.abs(self._wiring.sensory_adjacency_matrix)), requires_grad=False)
        if self._input_mapping in ['affine', 'linear']:
            self._params['input_w'] = self.add_weight(name='input_w', init_value=torch.ones((self.sensory_size,)))
        if self._input_mapping == 'affine':
            self._params['input_b'] = self.add_weight(name='input_b', init_value=torch.zeros((self.sensory_size,)))
        if self._output_mapping in ['affine', 'linear']:
            self._params['output_w'] = self.add_weight(name='output_w', init_value=torch.ones((self.motor_size,)))
        if self._output_mapping == 'affine':
            self._params['output_b'] = self.add_weight(name='output_b', init_value=torch.zeros((self.motor_size,)))

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = torch.unsqueeze(v_pre, -1)
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_solver(self, inputs, state, elapsed_time):
        v_pre = state
        sensory_w_activation = self.make_positive_fn(self._params['sensory_w']) * self._sigmoid(inputs, self._params['sensory_mu'], self._params['sensory_sigma'])
        sensory_w_activation = sensory_w_activation * self._params['sensory_sparsity_mask']
        sensory_rev_activation = sensory_w_activation * self._params['sensory_erev']
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)
        cm_t = self.make_positive_fn(self._params['cm']) / (elapsed_time / self._ode_unfolds)
        w_param = self.make_positive_fn(self._params['w'])
        for t in range(self._ode_unfolds):
            w_activation = w_param * self._sigmoid(v_pre, self._params['mu'], self._params['sigma'])
            w_activation = w_activation * self._params['sparsity_mask']
            rev_activation = w_activation * self._params['erev']
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory
            gleak = self.make_positive_fn(self._params['gleak'])
            numerator = cm_t * v_pre + gleak * self._params['vleak'] + w_numerator
            denominator = cm_t + gleak + w_denominator
            v_pre = numerator / (denominator + self._epsilon)
        return v_pre

    def _map_inputs(self, inputs):
        if self._input_mapping in ['affine', 'linear']:
            inputs = inputs * self._params['input_w']
        if self._input_mapping == 'affine':
            inputs = inputs + self._params['input_b']
        return inputs

    def _map_outputs(self, state):
        output = state
        if self.motor_size < self.state_size:
            output = output[:, 0:self.motor_size]
        if self._output_mapping in ['affine', 'linear']:
            output = output * self._params['output_w']
        if self._output_mapping == 'affine':
            output = output + self._params['output_b']
        return output

    def apply_weight_constraints(self):
        if not self._implicit_param_constraints:
            self._params['w'].data = self._clip(self._params['w'].data)
            self._params['sensory_w'].data = self._clip(self._params['sensory_w'].data)
            self._params['cm'].data = self._clip(self._params['cm'].data)
            self._params['gleak'].data = self._clip(self._params['gleak'].data)

    def forward(self, inputs, states, elapsed_time=1.0):
        inputs = self._map_inputs(inputs)
        next_state = self._ode_solver(inputs, states, elapsed_time)
        outputs = self._map_outputs(next_state)
        return outputs, next_state


class LTC(nn.Module):

    def __init__(self, input_size: int, units, return_sequences: bool=True, batch_first: bool=True, mixed_memory: bool=False, input_mapping='affine', output_mapping='affine', ode_unfolds=6, epsilon=1e-08, implicit_param_constraints=True):
        """Applies a `Liquid time-constant (LTC) <https://ojs.aaai.org/index.php/AAAI/article/view/16936>`_ RNN to an input sequence.

        Examples::

             >>> from ncps.torch import LTC
             >>>
             >>> rnn = LTC(20,50)
             >>> x = torch.randn(2, 3, 20) # (batch, time, features)
             >>> h0 = torch.zeros(2,50) # (batch, units)
             >>> output, hn = rnn(x,h0)

        .. Note::
            For creating a wired `Neural circuit policy (NCP) <https://publik.tuwien.ac.at/files/publik_292280.pdf>`_ you can pass a `ncps.wirings.NCP` object instead of the number of units

        Examples::

             >>> from ncps.torch import LTC
             >>> from ncps.wirings import NCP
             >>>
             >>> wiring = NCP(10, 10, 8, 6, 6, 4, 6)
             >>> rnn = LTC(20, wiring)

             >>> x = torch.randn(2, 3, 20) # (batch, time, features)
             >>> h0 = torch.zeros(2, 28) # (batch, units)
             >>> output, hn = rnn(x,h0)


        :param input_size: Number of input features
        :param units: Wiring (ncps.wirings.Wiring instance) or integer representing the number of (fully-connected) hidden units
        :param return_sequences: Whether to return the full sequence or just the last output
        :param batch_first: Whether the batch or time dimension is the first (0-th) dimension
        :param mixed_memory: Whether to augment the RNN by a `memory-cell <https://arxiv.org/abs/2006.04418>`_ to help learn long-term dependencies in the data
        :param input_mapping:
        :param output_mapping:
        :param ode_unfolds:
        :param epsilon:
        :param implicit_param_constraints:
        """
        super(LTC, self).__init__()
        self.input_size = input_size
        self.wiring_or_units = units
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        if isinstance(units, ncps.wirings.Wiring):
            wiring = units
        else:
            wiring = ncps.wirings.FullyConnected(units)
        self.rnn_cell = LTCCell(wiring=wiring, in_features=input_size, input_mapping=input_mapping, output_mapping=output_mapping, ode_unfolds=ode_unfolds, epsilon=epsilon, implicit_param_constraints=implicit_param_constraints)
        self._wiring = wiring
        self.use_mixed = mixed_memory
        if self.use_mixed:
            self.lstm = LSTMCell(input_size, self.state_size)

    @property
    def state_size(self):
        return self._wiring.units

    @property
    def sensory_size(self):
        return self._wiring.input_dim

    @property
    def motor_size(self):
        return self._wiring.output_dim

    @property
    def output_size(self):
        return self.motor_size

    @property
    def synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    def forward(self, input, hx=None, timespans=None):
        """

        :param input: Input tensor of shape (L,C) in batchless mode, or (B,L,C) if batch_first was set to True and (L,B,C) if batch_first is False
        :param hx: Initial hidden state of the RNN of shape (B,H) if mixed_memory is False and a tuple ((B,H),(B,H)) if mixed_memory is True. If None, the hidden states are initialized with all zeros.
        :param timespans:
        :return: A pair (output, hx), where output and hx the final hidden state of the RNN
        """
        device = input.device
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        if not is_batched:
            input = input.unsqueeze(batch_dim)
            if timespans is not None:
                timespans = timespans.unsqueeze(batch_dim)
        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)
        if hx is None:
            h_state = torch.zeros((batch_size, self.state_size), device=device)
            c_state = torch.zeros((batch_size, self.state_size), device=device) if self.use_mixed else None
        else:
            if self.use_mixed and isinstance(hx, torch.Tensor):
                raise RuntimeError('Running a CfC with mixed_memory=True, requires a tuple (h0,c0) to be passed as state (got torch.Tensor instead)')
            h_state, c_state = hx if self.use_mixed else (hx, None)
            if is_batched:
                if h_state.dim() != 2:
                    msg = f'For batched 2-D input, hx and cx should also be 2-D but got ({h_state.dim()}-D) tensor'
                    raise RuntimeError(msg)
            else:
                if h_state.dim() != 1:
                    msg = f'For unbatched 1-D input, hx and cx should also be 1-D but got ({h_state.dim()}-D) tensor'
                    raise RuntimeError(msg)
                h_state = h_state.unsqueeze(0)
                c_state = c_state.unsqueeze(0) if c_state is not None else None
        output_sequence = []
        for t in range(seq_len):
            if self.batch_first:
                inputs = input[:, t]
                ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            else:
                inputs = input[t]
                ts = 1.0 if timespans is None else timespans[t].squeeze()
            if self.use_mixed:
                h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_out, h_state = self.rnn_cell.forward(inputs, h_state, ts)
            if self.return_sequences:
                output_sequence.append(h_out)
        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = h_out
        hx = (h_state, c_state) if self.use_mixed else h_state
        if not is_batched:
            readout = readout.squeeze(batch_dim)
            hx = (h_state[0], c_state[0]) if self.use_mixed else h_state[0]
        return readout, hx


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CfCCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (ConvBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LeCun,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_mlech26l_ncps(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

