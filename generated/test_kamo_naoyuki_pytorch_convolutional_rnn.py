import sys
_module = sys.modules[__name__]
del sys
convolutional_rnn = _module
functional = _module
module = _module
utils = _module

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


from functools import partial


import torch


import torch.nn.functional as F


import math


from typing import Union


from typing import Sequence


from torch.nn import Parameter


from torch.nn.utils.rnn import PackedSequence


import collections


from itertools import repeat


def Recurrent(inner, reverse=False):
    """ Copied from torch.nn._functions.rnn without any modification """

    def forward(input, hidden, weight, batch_sizes):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            hidden = inner(input[i], hidden, *weight)
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)
        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        return hidden, output
    return forward


def StackedRNN(inners, num_layers, lstm=False, dropout=0, train=True):
    """ Copied from torch.nn._functions.rnn and modified """
    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight, batch_sizes):
        assert len(weight) == total_layers
        next_hidden = []
        ch_dim = input.dim() - weight[0][0].dim() + 1
        if lstm:
            hidden = list(zip(*hidden))
        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j
                hy, output = inner(input, hidden[l], weight[l], batch_sizes)
                next_hidden.append(hy)
                all_output.append(output)
            input = torch.cat(all_output, ch_dim)
            if dropout != 0 and i < num_layers - 1:
                input = F.dropout(input, p=dropout, training=train, inplace=False)
        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = torch.cat(next_h, 0).view(total_layers, *next_h[0].size()), torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
        else:
            next_hidden = torch.cat(next_hidden, 0).view(total_layers, *next_hidden[0].size())
        return next_hidden, input
    return forward


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_pair = _ntuple(2)


_single = _ntuple(1)


_triple = _ntuple(3)


def ConvNdWithSamePadding(convndim=2, stride=1, dilation=1, groups=1):

    def forward(input, w, b=None):
        if convndim == 1:
            ntuple = _single
        elif convndim == 2:
            ntuple = _pair
        elif convndim == 3:
            ntuple = _triple
        else:
            raise ValueError('convndim must be 1, 2, or 3, but got {}'.format(convndim))
        if input.dim() != convndim + 2:
            raise RuntimeError('Input dim must be {}, bot got {}'.format(convndim + 2, input.dim()))
        if w.dim() != convndim + 2:
            raise RuntimeError('w must be {}, bot got {}'.format(convndim + 2, w.dim()))
        insize = input.shape[2:]
        kernel_size = w.shape[2:]
        _stride = ntuple(stride)
        _dilation = ntuple(dilation)
        ps = [((i + 1 - h + s * (h - 1) + d * (k - 1)) // 2) for h, k, s, d in list(zip(insize, kernel_size, _stride, _dilation))[::-1] for i in range(2)]
        input = F.pad(input, ps, 'constant', 0)
        return getattr(F, 'conv{}d'.format(convndim))(input, w, b, stride=_stride, padding=ntuple(0), dilation=_dilation, groups=groups)
    return forward


def GRUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, linear_func=None):
    """ Copied from torch.nn._functions.rnn and modified """
    if linear_func is None:
        linear_func = F.linear
    if input.is_cuda and linear_func is F.linear and fusedBackend is not None:
        gi = linear_func(input, w_ih)
        gh = linear_func(hidden, w_hh)
        state = fusedBackend.GRUFused.apply
        return state(gi, gh, hidden) if b_ih is None else state(gi, gh, hidden, b_ih, b_hh)
    gi = linear_func(input, w_ih, b_ih)
    gh = linear_func(hidden, w_hh, b_hh)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)
    resetgate = torch.sigmoid(i_r + h_r)
    inputgate = torch.sigmoid(i_i + h_i)
    newgate = torch.tanh(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)
    return hy


def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, linear_func=None):
    """ Copied from torch.nn._functions.rnn and modified """
    if linear_func is None:
        linear_func = F.linear
    if input.is_cuda and linear_func is F.linear and fusedBackend is not None:
        igates = linear_func(input, w_ih)
        hgates = linear_func(hidden[0], w_hh)
        state = fusedBackend.LSTMFused.apply
        return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)
    hx, cx = hidden
    gates = linear_func(input, w_ih, b_ih) + linear_func(hx, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)
    cy = forgetgate * cx + ingate * cellgate
    hy = outgate * torch.tanh(cy)
    return hy, cy


def PeepholeLSTMCell(input, hidden, w_ih, w_hh, w_pi, w_pf, w_po, b_ih=None, b_hh=None, linear_func=None):
    if linear_func is None:
        linear_func = F.linear
    hx, cx = hidden
    gates = linear_func(input, w_ih, b_ih) + linear_func(hx, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate += linear_func(cx, w_pi)
    forgetgate += linear_func(cx, w_pf)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    cy = forgetgate * cx + ingate * cellgate
    outgate += linear_func(cy, w_po)
    outgate = torch.sigmoid(outgate)
    hy = outgate * torch.tanh(cy)
    return hy, cy


def RNNReLUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, linear_func=None):
    """ Copied from torch.nn._functions.rnn and modified """
    if linear_func is None:
        linear_func = F.linear
    hy = F.relu(linear_func(input, w_ih, b_ih) + linear_func(hidden, w_hh, b_hh))
    return hy


def RNNTanhCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, linear_func=None):
    """ Copied from torch.nn._functions.rnn and modified """
    if linear_func is None:
        linear_func = F.linear
    hy = torch.tanh(linear_func(input, w_ih, b_ih) + linear_func(hidden, w_hh, b_hh))
    return hy


def _conv_cell_helper(mode, convndim=2, stride=1, dilation=1, groups=1):
    linear_func = ConvNdWithSamePadding(convndim=convndim, stride=stride, dilation=dilation, groups=groups)
    if mode == 'RNN_RELU':
        cell = partial(RNNReLUCell, linear_func=linear_func)
    elif mode == 'RNN_TANH':
        cell = partial(RNNTanhCell, linear_func=linear_func)
    elif mode == 'LSTM':
        cell = partial(LSTMCell, linear_func=linear_func)
    elif mode == 'GRU':
        cell = partial(GRUCell, linear_func=linear_func)
    elif mode == 'PeepholeLSTM':
        cell = partial(PeepholeLSTMCell, linear_func=linear_func)
    else:
        raise Exception('Unknown mode: {}'.format(mode))
    return cell


def VariableRecurrent(inner):
    """ Copied from torch.nn._functions.rnn without any modification """

    def forward(input, hidden, weight, batch_sizes):
        output = []
        input_offset = 0
        last_batch_size = batch_sizes[0]
        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = hidden,
        for batch_size in batch_sizes:
            step_input = input[input_offset:input_offset + batch_size]
            input_offset += batch_size
            dec = last_batch_size - batch_size
            if dec > 0:
                hiddens.append(tuple(h[-dec:] for h in hidden))
                hidden = tuple(h[:-dec] for h in hidden)
            last_batch_size = batch_size
            if flat_hidden:
                hidden = inner(step_input, hidden[0], *weight),
            else:
                hidden = inner(step_input, hidden, *weight)
            output.append(hidden[0])
        hiddens.append(hidden)
        hiddens.reverse()
        hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
        assert hidden[0].size(0) == batch_sizes[0]
        if flat_hidden:
            hidden = hidden[0]
        output = torch.cat(output, 0)
        return hidden, output
    return forward


def VariableRecurrentReverse(inner):
    """ Copied from torch.nn._functions.rnn without any modification """

    def forward(input, hidden, weight, batch_sizes):
        output = []
        input_offset = input.size(0)
        last_batch_size = batch_sizes[-1]
        initial_hidden = hidden
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = hidden,
            initial_hidden = initial_hidden,
        hidden = tuple(h[:batch_sizes[-1]] for h in hidden)
        for i in reversed(range(len(batch_sizes))):
            batch_size = batch_sizes[i]
            inc = batch_size - last_batch_size
            if inc > 0:
                hidden = tuple(torch.cat((h, ih[last_batch_size:batch_size]), 0) for h, ih in zip(hidden, initial_hidden))
            last_batch_size = batch_size
            step_input = input[input_offset - batch_size:input_offset]
            input_offset -= batch_size
            if flat_hidden:
                hidden = inner(step_input, hidden[0], *weight),
            else:
                hidden = inner(step_input, hidden, *weight)
            output.append(hidden[0])
        output.reverse()
        output = torch.cat(output, 0)
        if flat_hidden:
            hidden = hidden[0]
        return hidden, output
    return forward


def variable_recurrent_factory(inner, reverse=False):
    """ Copied from torch.nn._functions.rnn without any modification """
    if reverse:
        return VariableRecurrentReverse(inner)
    else:
        return VariableRecurrent(inner)


def AutogradConvRNN(mode, num_layers=1, batch_first=False, dropout=0, train=True, bidirectional=False, variable_length=False, convndim=2, stride=1, dilation=1, groups=1):
    """ Copied from torch.nn._functions.rnn and modified """
    cell = _conv_cell_helper(mode, convndim=convndim, stride=stride, dilation=dilation, groups=groups)
    rec_factory = variable_recurrent_factory if variable_length else Recurrent
    if bidirectional:
        layer = rec_factory(cell), rec_factory(cell, reverse=True)
    else:
        layer = rec_factory(cell),
    func = StackedRNN(layer, num_layers, mode in ('LSTM', 'PeepholeLSTM'), dropout=dropout, train=train)

    def forward(input, weight, hidden, batch_sizes):
        if batch_first and batch_sizes is None:
            input = input.transpose(0, 1)
        nexth, output = func(input, hidden, weight, batch_sizes)
        if batch_first and batch_sizes is None:
            output = output.transpose(0, 1)
        return output, nexth
    return forward


class ConvNdRNNBase(torch.nn.Module):

    def __init__(self, mode: str, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], num_layers: int=1, bias: bool=True, batch_first: bool=False, dropout: float=0.0, bidirectional: bool=False, convndim: int=2, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.convndim = convndim
        if convndim == 1:
            ntuple = _single
        elif convndim == 2:
            ntuple = _pair
        elif convndim == 3:
            ntuple = _triple
        else:
            raise ValueError('convndim must be 1, 2, or 3, but got {}'.format(convndim))
        self.kernel_size = ntuple(kernel_size)
        self.stride = ntuple(stride)
        self.dilation = ntuple(dilation)
        self.groups = groups
        num_directions = 2 if bidirectional else 1
        if mode in ('LSTM', 'PeepholeLSTM'):
            gate_size = 4 * out_channels
        elif mode == 'GRU':
            gate_size = 3 * out_channels
        else:
            gate_size = out_channels
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = in_channels if layer == 0 else out_channels * num_directions
                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size // groups, *self.kernel_size))
                w_hh = Parameter(torch.Tensor(gate_size, out_channels // groups, *self.kernel_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))
                if mode == 'PeepholeLSTM':
                    w_pi = Parameter(torch.Tensor(out_channels, out_channels // groups, *self.kernel_size))
                    w_pf = Parameter(torch.Tensor(out_channels, out_channels // groups, *self.kernel_size))
                    w_po = Parameter(torch.Tensor(out_channels, out_channels // groups, *self.kernel_size))
                    layer_params = w_ih, w_hh, w_pi, w_pf, w_po, b_ih, b_hh
                    param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'weight_pi_l{}{}', 'weight_pf_l{}{}', 'weight_po_l{}{}']
                else:
                    layer_params = w_ih, w_hh, b_ih, b_hh
                    param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                suffix = '_reverse' if direction == 1 else ''
                param_names = [x.format(layer, suffix) for x in param_names]
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_channels)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = (2 if is_input_packed else 3) + self.convndim
        if input.dim() != expected_input_dim:
            raise RuntimeError('input must have {} dimensions, got {}'.format(expected_input_dim, input.dim()))
        ch_dim = 1 if is_input_packed else 2
        if self.in_channels != input.size(ch_dim):
            raise RuntimeError('input.size({}) must be equal to in_channels . Expected {}, got {}'.format(ch_dim, self.in_channels, input.size(ch_dim)))
        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions, mini_batch, self.out_channels) + input.shape[ch_dim + 1:]

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))
        if self.mode in ('LSTM', 'PeepholeLSTM'):
            check_hidden_size(hidden[0], expected_hidden_size, 'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size, 'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
            insize = input.shape[2:]
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            insize = input.shape[3:]
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(self.num_layers * num_directions, max_batch_size, self.out_channels, *insize, requires_grad=False)
            if self.mode in ('LSTM', 'PeepholeLSTM'):
                hx = hx, hx
        self.check_forward_args(input, hx, batch_sizes)
        func = AutogradConvRNN(self.mode, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout, train=self.training, bidirectional=self.bidirectional, variable_length=batch_sizes is not None, convndim=self.convndim, stride=self.stride, dilation=self.dilation, groups=self.groups)
        output, hidden = func(input, self.all_weights, hx, batch_sizes)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(ConvNdRNNBase, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                if self.mode == 'PeepholeLSTM':
                    weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'weight_pi_l{}{}', 'weight_pf_l{}{}', 'weight_po_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                else:
                    weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:len(weights) // 2]]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]


class Conv1dRNN(ConvNdRNNBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], nonlinearity: str='tanh', num_layers: int=1, bias: bool=True, batch_first: bool=False, dropout: float=0.0, bidirectional: bool=False, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        if nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        super().__init__(mode=mode, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, convndim=1, stride=stride, dilation=dilation, groups=groups)


class Conv1dPeepholeLSTM(ConvNdRNNBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], num_layers: int=1, bias: bool=True, batch_first: bool=False, dropout: float=0.0, bidirectional: bool=False, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__(mode='PeepholeLSTM', in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, convndim=1, stride=stride, dilation=dilation, groups=groups)


class Conv1dLSTM(ConvNdRNNBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], num_layers: int=1, bias: bool=True, batch_first: bool=False, dropout: float=0.0, bidirectional: bool=False, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__(mode='LSTM', in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, convndim=1, stride=stride, dilation=dilation, groups=groups)


class Conv1dGRU(ConvNdRNNBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], num_layers: int=1, bias: bool=True, batch_first: bool=False, dropout: float=0.0, bidirectional: bool=False, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__(mode='GRU', in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, convndim=1, stride=stride, dilation=dilation, groups=groups)


class Conv2dRNN(ConvNdRNNBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], nonlinearity: str='tanh', num_layers: int=1, bias: bool=True, batch_first: bool=False, dropout: float=0.0, bidirectional: bool=False, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        if nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        super().__init__(mode=mode, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, convndim=2, stride=stride, dilation=dilation, groups=groups)


class Conv2dLSTM(ConvNdRNNBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], num_layers: int=1, bias: bool=True, batch_first: bool=False, dropout: float=0.0, bidirectional: bool=False, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__(mode='LSTM', in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, convndim=2, stride=stride, dilation=dilation, groups=groups)


class Conv2dPeepholeLSTM(ConvNdRNNBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], num_layers: int=1, bias: bool=True, batch_first: bool=False, dropout: float=0.0, bidirectional: bool=False, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__(mode='PeepholeLSTM', in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, convndim=2, stride=stride, dilation=dilation, groups=groups)


class Conv2dGRU(ConvNdRNNBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], num_layers: int=1, bias: bool=True, batch_first: bool=False, dropout: float=0.0, bidirectional: bool=False, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__(mode='GRU', in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, convndim=2, stride=stride, dilation=dilation, groups=groups)


class Conv3dRNN(ConvNdRNNBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], nonlinearity: str='tanh', num_layers: int=1, bias: bool=True, batch_first: bool=False, dropout: float=0.0, bidirectional: bool=False, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        if nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        super().__init__(mode=mode, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, convndim=3, stride=stride, dilation=dilation, groups=groups)


class Conv3dLSTM(ConvNdRNNBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], num_layers: int=1, bias: bool=True, batch_first: bool=False, dropout: float=0.0, bidirectional: bool=False, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__(mode='LSTM', in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, convndim=3, stride=stride, dilation=dilation, groups=groups)


class Conv3dPeepholeLSTM(ConvNdRNNBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], num_layers: int=1, bias: bool=True, batch_first: bool=False, dropout: float=0.0, bidirectional: bool=False, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__(mode='PeepholeLSTM', in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, convndim=3, stride=stride, dilation=dilation, groups=groups)


class Conv3dGRU(ConvNdRNNBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], num_layers: int=1, bias: bool=True, batch_first: bool=False, dropout: float=0.0, bidirectional: bool=False, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__(mode='GRU', in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, convndim=3, stride=stride, dilation=dilation, groups=groups)


class ConvRNNCellBase(torch.nn.Module):

    def __init__(self, mode: str, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], bias: bool=True, convndim: int=2, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.convndim = convndim
        if convndim == 1:
            ntuple = _single
        elif convndim == 2:
            ntuple = _pair
        elif convndim == 3:
            ntuple = _triple
        else:
            raise ValueError('convndim must be 1, 2, or 3, but got {}'.format(convndim))
        self.kernel_size = ntuple(kernel_size)
        self.stride = ntuple(stride)
        self.dilation = ntuple(dilation)
        self.groups = groups
        if mode in ('LSTM', 'PeepholeLSTM'):
            gate_size = 4 * out_channels
        elif mode == 'GRU':
            gate_size = 3 * out_channels
        else:
            gate_size = out_channels
        self.weight_ih = Parameter(torch.Tensor(gate_size, in_channels // groups, *self.kernel_size))
        self.weight_hh = Parameter(torch.Tensor(gate_size, out_channels // groups, *self.kernel_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(gate_size))
            self.bias_hh = Parameter(torch.Tensor(gate_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        if mode == 'PeepholeLSTM':
            self.weight_pi = Parameter(torch.Tensor(out_channels, out_channels // groups, *self.kernel_size))
            self.weight_pf = Parameter(torch.Tensor(out_channels, out_channels // groups, *self.kernel_size))
            self.weight_po = Parameter(torch.Tensor(out_channels, out_channels // groups, *self.kernel_size))
        self.reset_parameters()

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.in_channels:
            raise RuntimeError('input has inconsistent channels: got {}, expected {}'.format(input.size(1), self.in_channels))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError("Input batch size {} doesn't match hidden{} batch size {}".format(input.size(0), hidden_label, hx.size(0)))
        if hx.size(1) != self.out_channels:
            raise RuntimeError('hidden{} has inconsistent hidden_size: got {}, expected {}'.format(hidden_label, hx.size(1), self.out_channels))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_channels)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        self.check_forward_input(input)
        if hx is None:
            batch_size = input.size(0)
            insize = input.shape[2:]
            hx = input.new_zeros(batch_size, self.out_channels, *insize, requires_grad=False)
            if self.mode in ('LSTM', 'PeepholeLSTM'):
                hx = hx, hx
        if self.mode in ('LSTM', 'PeepholeLSTM'):
            self.check_forward_hidden(input, hx[0])
            self.check_forward_hidden(input, hx[1])
        else:
            self.check_forward_hidden(input, hx)
        cell = _conv_cell_helper(self.mode, convndim=self.convndim, stride=self.stride, dilation=self.dilation, groups=self.groups)
        if self.mode == 'PeepholeLSTM':
            return cell(input, hx, self.weight_ih, self.weight_hh, self.weight_pi, self.weight_pf, self.weight_po, self.bias_ih, self.bias_hh)
        else:
            return cell(input, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)


class Conv1dRNNCell(ConvRNNCellBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], nonlinearity: str='tanh', bias: bool=True, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        if nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        super().__init__(mode=mode, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, convndim=1, stride=stride, dilation=dilation, groups=groups)


class Conv1dLSTMCell(ConvRNNCellBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], bias: bool=True, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__(mode='LSTM', in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, convndim=1, stride=stride, dilation=dilation, groups=groups)


class Conv1dPeepholeLSTMCell(ConvRNNCellBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], bias: bool=True, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__(mode='PeepholeLSTM', in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, convndim=1, stride=stride, dilation=dilation, groups=groups)


class Conv1dGRUCell(ConvRNNCellBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], bias: bool=True, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__(mode='GRU', in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, convndim=1, stride=stride, dilation=dilation, groups=groups)


class Conv2dRNNCell(ConvRNNCellBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], nonlinearity: str='tanh', bias: bool=True, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        if nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        super().__init__(mode=mode, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, convndim=2, stride=stride, dilation=dilation, groups=groups)


class Conv2dLSTMCell(ConvRNNCellBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], bias: bool=True, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__(mode='LSTM', in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, convndim=2, stride=stride, dilation=dilation, groups=groups)


class Conv2dPeepholeLSTMCell(ConvRNNCellBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], bias: bool=True, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__(mode='PeepholeLSTM', in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, convndim=2, stride=stride, dilation=dilation, groups=groups)


class Conv2dGRUCell(ConvRNNCellBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], bias: bool=True, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__(mode='GRU', in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, convndim=2, stride=stride, dilation=dilation, groups=groups)


class Conv3dRNNCell(ConvRNNCellBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], nonlinearity: str='tanh', bias: bool=True, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        if nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        super().__init__(mode=mode, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, convndim=3, stride=stride, dilation=dilation, groups=groups)


class Conv3dLSTMCell(ConvRNNCellBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], bias: bool=True, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__(mode='LSTM', in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, convndim=3, stride=stride, dilation=dilation, groups=groups)


class Conv3dPeepholeLSTMCell(ConvRNNCellBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], bias: bool=True, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__(mode='PeepholeLSTM', in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, convndim=3, stride=stride, dilation=dilation, groups=groups)


class Conv3dGRUCell(ConvRNNCellBase):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Sequence[int]], bias: bool=True, stride: Union[int, Sequence[int]]=1, dilation: Union[int, Sequence[int]]=1, groups: int=1):
        super().__init__(mode='GRU', in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, convndim=3, stride=stride, dilation=dilation, groups=groups)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv1dGRU,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv1dGRUCell,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Conv1dLSTM,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv1dLSTMCell,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Conv1dPeepholeLSTM,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv1dPeepholeLSTMCell,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Conv1dRNN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv1dRNNCell,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Conv2dGRU,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (Conv2dGRUCell,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv2dLSTM,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (Conv2dLSTMCell,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv2dPeepholeLSTM,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (Conv2dPeepholeLSTMCell,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv2dRNN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (Conv2dRNNCell,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv3dGRU,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4, 4])], {}),
     False),
    (Conv3dLSTM,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4, 4])], {}),
     False),
    (Conv3dPeepholeLSTM,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4, 4])], {}),
     False),
    (Conv3dRNN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4, 4])], {}),
     False),
]

class Test_kamo_naoyuki_pytorch_convolutional_rnn(_paritybench_base):
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

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

