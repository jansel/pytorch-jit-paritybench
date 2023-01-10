import sys
_module = sys.modules[__name__]
del sys
conf = _module
libreasr_pb2 = _module
libreasr_pb2_grpc = _module
libreasr = _module
lib = _module
builder = _module
callbacks = _module
config = _module
data = _module
decoders = _module
imports = _module
inference = _module
inference_imports = _module
language = _module
layers = _module
custom_rnn = _module
haste = _module
base_rnn = _module
gru = _module
layer_norm_lstm = _module
lstm = _module
nbrc = _module
mish = _module
learner = _module
lm = _module
loss = _module
metrics = _module
model_utils = _module
models = _module
optimizer = _module
patches = _module
transforms = _module
utils = _module
split = _module

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


import time


import math


import logging


import itertools as it


from functools import partial


import pandas as pd


import numpy as np


import torch


import torchaudio


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter


import collections.abc


import random


from typing import Tuple


from itertools import groupby


from collections import OrderedDict


import torch.quantization


import string


from torch.nn import Parameter


from torch.nn import ParameterList


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from queue import PriorityQueue


from torch import nn


from torch.optim.optimizer import Optimizer


from scipy.signal import decimate


from scipy.signal import resample_poly


import re


import matplotlib.pyplot as plt


def get_initial_state(rnn_type, hidden_size, init=torch.zeros):
    if rnn_type == 'LSTM':
        h = nn.Parameter(init(2, 1, 1, hidden_size))
        tmp = init(2, 1, 1, hidden_size)
    else:
        h = nn.Parameter(init(1, 1, 1, hidden_size))
        tmp = init(1, 1, 1, hidden_size)
    return h, tmp


class CustomRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, rnn_type='LSTM', reduction_indices=[], reduction_factors=[], reduction_drop=True, rezero=False, layer_norm=False, utsp=0.9):
        super().__init__()
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self._is = [input_size] + [hidden_size] * (num_layers - 1)
        self._os = [hidden_size] * num_layers
        self.rnn_type = rnn_type
        assert len(reduction_indices) == len(reduction_factors)
        self.reduction_indices = reduction_indices
        self.reduction_factors = reduction_factors
        self.hs = nn.ParameterList()
        for hidden_size in self._os:
            h, tmp = get_initial_state(rnn_type, hidden_size)
            self.hs.append(h)
        self.cache = {}
        self.bns = nn.ModuleList()
        for i, o in zip(self._is, self._os):
            norm = nn.BatchNorm1d(o)
            self.bns.append(norm)
        self.rezero = rezero
        self.utsp = utsp

    def convert_to_cpu(self):
        return self

    def convert_to_gpu(self):
        return self

    def forward_one_rnn(self, x, i, state=None, should_use_tmp_state=False, lengths=None):
        bs = x.size(0)
        if state is None:
            s = self.cache[bs][i] if self.cache.get(bs) is not None else None
            is_tmp_state_possible = self.training and s is not None
            if is_tmp_state_possible and should_use_tmp_state:
                pass
            elif self.hs[i].size(0) == 2:
                s = []
                for h in self.hs[i]:
                    s.append(h.expand(1, bs, self._os[i]).contiguous())
                s = tuple(s)
            else:
                s = self.hs[i][0].expand(1, bs, self._os[i]).contiguous()
        else:
            s = state[i]
        if self.rnn_type == 'LSTM' or self.rnn_type == 'GRU':
            if lengths is not None:
                seq = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
                seq, new_state = self.rnns[i](seq, s)
                x, _ = pad_packed_sequence(seq, batch_first=True)
                return x, new_state
            else:
                return self.rnns[i](x, s)
        else:
            return self.rnns[i](x, s, lengths=lengths if lengths is not None else None)

    def forward(self, x, state=None, lengths=None):
        bs = x.size(0)
        residual = 0.0
        new_states = []
        suts = random.random() > 1.0 - self.utsp
        for i, rnn in enumerate(self.rnns):
            if i in self.reduction_indices:
                idx = self.reduction_indices.index(i)
                r_f = self.reduction_factors[idx]
                x = x.permute(0, 2, 1)
                x = x.unfold(-1, r_f, r_f)
                x = x.permute(0, 2, 1, 3)
                x = x.mean(-1)
                if lengths is not None:
                    lengths = lengths // r_f
            inp = x
            x, new_state = self.forward_one_rnn(inp, i, state=state, should_use_tmp_state=suts, lengths=lengths)
            x = x.permute(0, 2, 1)
            x = self.bns[i](x)
            x = x.permute(0, 2, 1)
            if self.rezero:
                if torch.is_tensor(residual) and residual.shape == x.shape:
                    x = x + residual
            residual = inp
            new_states.append(new_state)
        if self.training:
            if len(new_states[0]) == 2:
                self.cache[bs] = [(h.detach().contiguous(), c.detach().contiguous()) for h, c in new_states]
            else:
                self.cache[bs] = [h.detach() for h in new_states]
        return x, new_states


USE_PYTORCH = True


def copy_weights(_from, _to, attrs):
    for attr in attrs:
        setattr(_to, attr, getattr(_from, attr))


DEVICES = ['CPU', 'GPU']


RNN_TYPES = ['LSTM', 'GRU', 'NBRC']


def get_weight_attrs(rnn_type, layer_norm):
    attrs = ['kernel', 'recurrent_kernel', 'bias']
    if rnn_type == 'GRU' or rnn_type == 'NBRC':
        attrs += ['recurrent_bias']
    if layer_norm:
        attrs += ['gamma', 'gamma_h', 'beta_h']
    return attrs


class CustomCPURNN(CustomRNN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._args = args
        self._kwargs = kwargs
        RNN = get_rnn_impl('CPU', self.rnn_type, kwargs['layer_norm'])
        self.rnns = nn.ModuleList()
        for i, o in zip(self._is, self._os):
            r = RNN(i, o, batch_first=self.batch_first)
            self.rnns.append(r)

    def convert_to_gpu(self):
        dev = next(self.parameters()).device
        if USE_PYTORCH or self.rnn_type == 'NBRC':
            return self
        inst = CustomGPURNN(*self._args, **self._kwargs)
        attrs = get_weight_attrs(self.rnn_type, self._kwargs['layer_norm'])
        for i, rnn in enumerate(self.rnns):
            grabbed_rnn = inst.rnns[i]
            copy_weights(rnn, grabbed_rnn, attrs)
        return inst


class CustomGPURNN(CustomRNN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._args = args
        self._kwargs = kwargs
        RNN = get_rnn_impl('GPU', self.rnn_type, kwargs['layer_norm'])
        self.rnns = nn.ModuleList()
        for i, o in zip(self._is, self._os):
            r = RNN(i, o, batch_first=self.batch_first)
            self.rnns.append(r)

    def convert_to_cpu(self):
        if USE_PYTORCH:
            return self
        dev = next(self.parameters()).device
        inst = CustomCPURNN(*self._args, **self._kwargs)
        attrs = get_weight_attrs(self.rnn_type, self._kwargs['layer_norm'])
        for i, rnn in enumerate(self.rnns):
            grabbed_rnn = inst.rnns[i]
            copy_weights(rnn, grabbed_rnn, attrs)
        return inst


def _validate_state(state, state_shape):
    """
  Checks to make sure that `state` has the same nested structure and dimensions
  as `state_shape`. `None` values in `state_shape` are a wildcard and are not
  checked.

  Arguments:
    state: a nested structure of Tensors.
    state_shape: a nested structure of integers or None.

  Raises:
    ValueError: if the structure and/or shapes don't match.
  """
    if isinstance(state, (tuple, list)):
        if not isinstance(state_shape, (tuple, list)):
            raise ValueError('RNN state has invalid structure; expected {}'.format(state_shape))
        for s, ss in zip(state, state_shape):
            _validate_state(s, ss)
    else:
        shape = list(state.size())
        if len(shape) != len(state_shape):
            raise ValueError('RNN state dimension mismatch; expected {} got {}'.format(len(state_shape), len(shape)))
        for i, (d1, d2) in enumerate(zip(list(state.size()), state_shape)):
            if d2 is not None and d1 != d2:
                raise ValueError('RNN state size mismatch on dim {}; expected {} got {}'.format(i, d2, d1))


def _zero_state(input, state_shape):
    """
  Returns a nested structure of zero Tensors with the same structure and shape
  as `state_shape`. The returned Tensors will have the same dtype and be on the
  same device as `input`.

  Arguments:
    input: Tensor, to specify the device and dtype of the returned tensors.
    shape_state: nested structure of integers.

  Returns:
    zero_state: a nested structure of zero Tensors.

  Raises:
    ValueError: if `state_shape` has non-integer values.
  """
    if isinstance(state_shape, (tuple, list)) and isinstance(state_shape[0], int):
        state = input.new_zeros(*state_shape)
    elif isinstance(state_shape, tuple):
        state = tuple(_zero_state(input, s) for s in state_shape)
    elif isinstance(state_shape, list):
        state = [_zero_state(input, s) for s in state_shape]
    else:
        raise ValueError('RNN state_shape is invalid')
    return state


class BaseRNN(nn.Module):

    def __init__(self, input_size, hidden_size, batch_first, zoneout, return_state_sequence):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.zoneout = zoneout
        self.return_state_sequence = return_state_sequence

    def _permute(self, x):
        if self.batch_first:
            return x.permute(1, 0, 2)
        return x

    def _get_state(self, input, state, state_shape):
        if state is None:
            state = _zero_state(input, state_shape)
        else:
            _validate_state(state, state_shape)
        return state

    def _get_final_state(self, state, lengths):
        if isinstance(state, tuple):
            return tuple(self._get_final_state(s, lengths) for s in state)
        if isinstance(state, list):
            return [self._get_final_state(s, lengths) for s in state]
        if self.return_state_sequence:
            return self._permute(state[1:]).unsqueeze(0)
        if lengths is not None:
            cols = range(state.size(1))
            return state[[lengths, cols]].unsqueeze(0)
        return state[-1].unsqueeze(0)

    def _get_zoneout_mask(self, input):
        if self.zoneout:
            zoneout_mask = input.new_empty(input.shape[0], input.shape[1], self.hidden_size)
            zoneout_mask.bernoulli_(1.0 - self.zoneout)
        else:
            zoneout_mask = input.new_empty(0, 0, 0)
        return zoneout_mask

    def _is_cuda(self):
        is_cuda = [tensor.is_cuda for tensor in list(self.parameters())]
        if any(is_cuda) and not all(is_cuda):
            raise ValueError('RNN tensors should all be CUDA tensors or none should be CUDA tensors')
        return any(is_cuda)


def GRUScript(training: bool, zoneout_prob: float, input, h0, kernel, recurrent_kernel, bias, recurrent_bias, zoneout_mask):
    time_steps = input.shape[0]
    batch_size = input.shape[1]
    hidden_size = recurrent_kernel.shape[0]
    h = [h0]
    Wx = input @ kernel + bias
    for t in range(time_steps):
        Rh = h[t] @ recurrent_kernel + recurrent_bias
        vx = torch.chunk(Wx[t], 3, 1)
        vh = torch.chunk(Rh, 3, 1)
        z = torch.sigmoid(vx[0] + vh[0])
        r = torch.sigmoid(vx[1] + vh[1])
        g = torch.tanh(vx[2] + r * vh[2])
        h.append(z * h[t] + (1 - z) * g)
        if zoneout_prob:
            if training:
                h[-1] = (h[-1] - h[-2]) * zoneout_mask[t] + h[-2]
            else:
                h[-1] = zoneout_prob * h[-2] + (1 - zoneout_prob) * h[-1]
    h = torch.stack(h)
    return h


class GRU(BaseRNN):
    """
  Gated Recurrent Unit layer.

  This GRU layer offers a fused, GPU-accelerated PyTorch op for inference
  and training. There are two commonly-used variants of GRU cells. This one
  implements 1406.1078v1 which applies the reset gate to the hidden state
  after matrix multiplication. cuDNN also implements this variant. The other
  variant, 1406.1078v3, applies the reset gate before matrix multiplication
  and is currently unsupported.

  This layer has built-in support for DropConnect and Zoneout, which are
  both techniques used to regularize RNNs.

  See [\\_\\_init\\_\\_](#__init__) and [forward](#forward) for usage.
  See [from_native_weights](#from_native_weights) and
  [to_native_weights](#to_native_weights) for compatibility with PyTorch GRUs.
  """

    def __init__(self, input_size, hidden_size, batch_first=False, dropout=0.0, zoneout=0.0, return_state_sequence=False):
        """
    Initialize the parameters of the GRU layer.

    Arguments:
      input_size: int, the feature dimension of the input.
      hidden_size: int, the feature dimension of the output.
      batch_first: (optional) bool, if `True`, then the input and output
        tensors are provided as `(batch, seq, feature)`.
      dropout: (optional) float, sets the dropout rate for DropConnect
        regularization on the recurrent matrix.
      zoneout: (optional) float, sets the zoneout rate for Zoneout
        regularization.
      return_state_sequence: (optional) bool, if `True`, the forward pass will
        return the entire state sequence instead of just the final state. Note
        that if the input is a padded sequence, the returned state will also
        be a padded sequence.

    Variables:
      kernel: the input projection weight matrix. Dimensions
        (input_size, hidden_size * 3) with `z,r,h` gate layout. Initialized
        with Xavier uniform initialization.
      recurrent_kernel: the recurrent projection weight matrix. Dimensions
        (hidden_size, hidden_size * 3) with `z,r,h` gate layout. Initialized
        with orthogonal initialization.
      bias: the input projection bias vector. Dimensions (hidden_size * 3) with
        `z,r,h` gate layout. Initialized to zeros.
      recurrent_bias: the recurrent projection bias vector. Dimensions
        (hidden_size * 3) with `z,r,h` gate layout. Initialized to zeros.
    """
        super().__init__(input_size, hidden_size, batch_first, zoneout, return_state_sequence)
        if dropout < 0 or dropout > 1:
            raise ValueError('GRU: dropout must be in [0.0, 1.0]')
        if zoneout < 0 or zoneout > 1:
            raise ValueError('GRU: zoneout must be in [0.0, 1.0]')
        self.dropout = dropout
        self.kernel = nn.Parameter(torch.empty(input_size, hidden_size * 3))
        self.recurrent_kernel = nn.Parameter(torch.empty(hidden_size, hidden_size * 3))
        self.bias = nn.Parameter(torch.empty(hidden_size * 3))
        self.recurrent_bias = nn.Parameter(torch.empty(hidden_size * 3))
        self.reset_parameters()

    def to_native_weights(self):
        """
    Converts Haste GRU weights to native PyTorch GRU weights.

    Returns:
      weight_ih_l0: Parameter, the input-hidden weights of the GRU layer.
      weight_hh_l0: Parameter, the hidden-hidden weights of the GRU layer.
      bias_ih_l0: Parameter, the input-hidden bias of the GRU layer.
      bias_hh_l0: Parameter, the hidden-hidden bias of the GRU layer.
    """

        def reorder_weights(w):
            z, r, n = torch.chunk(w, 3, dim=-1)
            return torch.cat([r, z, n], dim=-1)
        kernel = reorder_weights(self.kernel).permute(1, 0).contiguous()
        recurrent_kernel = reorder_weights(self.recurrent_kernel).permute(1, 0).contiguous()
        bias1 = reorder_weights(self.bias).contiguous()
        bias2 = reorder_weights(self.recurrent_bias).contiguous()
        kernel = torch.nn.Parameter(kernel)
        recurrent_kernel = torch.nn.Parameter(recurrent_kernel)
        bias1 = torch.nn.Parameter(bias1)
        bias2 = torch.nn.Parameter(bias2)
        return kernel, recurrent_kernel, bias1, bias2

    def from_native_weights(self, weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0):
        """
    Copies and converts the provided PyTorch GRU weights into this layer.

    Arguments:
      weight_ih_l0: Parameter, the input-hidden weights of the PyTorch GRU layer.
      weight_hh_l0: Parameter, the hidden-hidden weights of the PyTorch GRU layer.
      bias_ih_l0: Parameter, the input-hidden bias of the PyTorch GRU layer.
      bias_hh_l0: Parameter, the hidden-hidden bias of the PyTorch GRU layer.
    """

        def reorder_weights(w):
            r, z, n = torch.chunk(w, 3, axis=-1)
            return torch.cat([z, r, n], dim=-1)
        kernel = reorder_weights(weight_ih_l0.permute(1, 0)).contiguous()
        recurrent_kernel = reorder_weights(weight_hh_l0.permute(1, 0)).contiguous()
        bias = reorder_weights(bias_ih_l0).contiguous()
        recurrent_bias = reorder_weights(bias_hh_l0).contiguous()
        self.kernel = nn.Parameter(kernel)
        self.recurrent_kernel = nn.Parameter(recurrent_kernel)
        self.bias = nn.Parameter(bias)
        self.recurrent_bias = nn.Parameter(recurrent_bias)

    def reset_parameters(self):
        """Resets this layer's parameters to their initial values."""
        hidden_size = self.hidden_size
        for i in range(3):
            nn.init.xavier_uniform_(self.kernel[:, i * hidden_size:(i + 1) * hidden_size])
            nn.init.orthogonal_(self.recurrent_kernel[:, i * hidden_size:(i + 1) * hidden_size])
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.recurrent_bias)

    def forward(self, input, state=None, lengths=None):
        """
    Runs a forward pass of the GRU layer.

    Arguments:
      input: Tensor, a batch of input sequences to pass through the GRU.
        Dimensions (seq_len, batch_size, input_size) if `batch_first` is
        `False`, otherwise (batch_size, seq_len, input_size).
      lengths: (optional) Tensor, list of sequence lengths for each batch
        element. Dimension (batch_size). This argument may be omitted if
        all batch elements are unpadded and have the same sequence length.

    Returns:
      output: Tensor, the output of the GRU layer. Dimensions
        (seq_len, batch_size, hidden_size) if `batch_first` is `False` (default)
        or (batch_size, seq_len, hidden_size) if `batch_first` is `True`. Note
        that if `lengths` was specified, the `output` tensor will not be
        masked. It's the caller's responsibility to either not use the invalid
        entries or to mask them out before using them.
      h_n: the hidden state for the last sequence item. Dimensions
        (1, batch_size, hidden_size).
    """
        input = self._permute(input)
        state_shape = [1, input.shape[1], self.hidden_size]
        h0 = self._get_state(input, state, state_shape)
        h = self._impl(input, h0[0], self._get_zoneout_mask(input))
        state = self._get_final_state(h, lengths)
        output = self._permute(h[1:])
        return output, state

    def _impl(self, input, state, zoneout_mask):
        return GRUScript(self.training, self.zoneout, input.contiguous(), state.contiguous(), self.kernel.contiguous(), F.dropout(self.recurrent_kernel, self.dropout, self.training).contiguous(), self.bias.contiguous(), self.recurrent_bias.contiguous(), zoneout_mask.contiguous())


def LayerNormLSTMScript(training: bool, zoneout_prob: float, input, h0, c0, kernel, recurrent_kernel, bias, gamma, gamma_h, beta_h, zoneout_mask):
    time_steps = input.shape[0]
    batch_size = input.shape[1]
    hidden_size = recurrent_kernel.shape[0]
    h = [h0]
    c = [c0]
    Wx = F.layer_norm(input @ kernel, (hidden_size * 4,), weight=gamma[0])
    for t in range(time_steps):
        v = F.layer_norm(h[t] @ recurrent_kernel, (hidden_size * 4,), weight=gamma[1]) + Wx[t] + bias
        i, g, f, o = torch.chunk(v, 4, 1)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        c.append(f * c[t] + i * g)
        h.append(o * torch.tanh(F.layer_norm(c[-1], (hidden_size,), weight=gamma_h, bias=beta_h)))
        if zoneout_prob:
            if training:
                h[-1] = (h[-1] - h[-2]) * zoneout_mask[t] + h[-2]
            else:
                h[-1] = zoneout_prob * h[-2] + (1 - zoneout_prob) * h[-1]
    h = torch.stack(h)
    c = torch.stack(c)
    return h, c


class LayerNormLSTM(BaseRNN):
    """
  Layer Normalized Long Short-Term Memory layer.

  This LSTM layer applies layer normalization to the input, recurrent, and
  output activations of a standard LSTM. The implementation is fused and
  GPU-accelerated. DropConnect and Zoneout regularization are built-in, and
  this layer allows setting a non-zero initial forget gate bias.

  Details about the exact function this layer implements can be found at
  https://github.com/lmnt-com/haste/issues/1.

  See [\\_\\_init\\_\\_](#__init__) and [forward](#forward) for usage.
  """

    def __init__(self, input_size, hidden_size, batch_first=False, forget_bias=1.0, dropout=0.0, zoneout=0.0, return_state_sequence=False):
        """
    Initialize the parameters of the LSTM layer.

    Arguments:
      input_size: int, the feature dimension of the input.
      hidden_size: int, the feature dimension of the output.
      batch_first: (optional) bool, if `True`, then the input and output
        tensors are provided as `(batch, seq, feature)`.
      forget_bias: (optional) float, sets the initial bias of the forget gate
        for this LSTM cell.
      dropout: (optional) float, sets the dropout rate for DropConnect
        regularization on the recurrent matrix.
      zoneout: (optional) float, sets the zoneout rate for Zoneout
        regularization.
      return_state_sequence: (optional) bool, if `True`, the forward pass will
        return the entire state sequence instead of just the final state. Note
        that if the input is a padded sequence, the returned state will also
        be a padded sequence.

    Variables:
      kernel: the input projection weight matrix. Dimensions
        (input_size, hidden_size * 4) with `i,g,f,o` gate layout. Initialized
        with Xavier uniform initialization.
      recurrent_kernel: the recurrent projection weight matrix. Dimensions
        (hidden_size, hidden_size * 4) with `i,g,f,o` gate layout. Initialized
        with orthogonal initialization.
      bias: the projection bias vector. Dimensions (hidden_size * 4) with
        `i,g,f,o` gate layout. The forget gate biases are initialized to
        `forget_bias` and the rest are zeros.
      gamma: the input and recurrent normalization gain. Dimensions
        (2, hidden_size * 4) with `gamma[0]` specifying the input gain and
        `gamma[1]` specifying the recurrent gain. Initialized to ones.
      gamma_h: the output normalization gain. Dimensions (hidden_size).
        Initialized to ones.
      beta_h: the output normalization bias. Dimensions (hidden_size).
        Initialized to zeros.
    """
        super().__init__(input_size, hidden_size, batch_first, zoneout, return_state_sequence)
        if dropout < 0 or dropout > 1:
            raise ValueError('LayerNormLSTM: dropout must be in [0.0, 1.0]')
        if zoneout < 0 or zoneout > 1:
            raise ValueError('LayerNormLSTM: zoneout must be in [0.0, 1.0]')
        self.forget_bias = forget_bias
        self.dropout = dropout
        self.kernel = nn.Parameter(torch.empty(input_size, hidden_size * 4))
        self.recurrent_kernel = nn.Parameter(torch.empty(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.empty(hidden_size * 4))
        self.gamma = nn.Parameter(torch.empty(2, hidden_size * 4))
        self.gamma_h = nn.Parameter(torch.empty(hidden_size))
        self.beta_h = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        """Resets this layer's parameters to their initial values."""
        hidden_size = self.hidden_size
        for i in range(4):
            nn.init.xavier_uniform_(self.kernel[:, i * hidden_size:(i + 1) * hidden_size])
            nn.init.orthogonal_(self.recurrent_kernel[:, i * hidden_size:(i + 1) * hidden_size])
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.bias[hidden_size * 2:hidden_size * 3], self.forget_bias)
        nn.init.ones_(self.gamma)
        nn.init.ones_(self.gamma_h)
        nn.init.zeros_(self.beta_h)

    def forward(self, input, state=None, lengths=None):
        """
    Runs a forward pass of the LSTM layer.

    Arguments:
      input: Tensor, a batch of input sequences to pass through the LSTM.
        Dimensions (seq_len, batch_size, input_size) if `batch_first` is
        `False`, otherwise (batch_size, seq_len, input_size).
      lengths: (optional) Tensor, list of sequence lengths for each batch
        element. Dimension (batch_size). This argument may be omitted if
        all batch elements are unpadded and have the same sequence length.

    Returns:
      output: Tensor, the output of the LSTM layer. Dimensions
        (seq_len, batch_size, hidden_size) if `batch_first` is `False` (default)
        or (batch_size, seq_len, hidden_size) if `batch_first` is `True`. Note
        that if `lengths` was specified, the `output` tensor will not be
        masked. It's the caller's responsibility to either not use the invalid
        entries or to mask them out before using them.
      (h_n, c_n): the hidden and cell states, respectively, for the last
        sequence item. Dimensions (1, batch_size, hidden_size).
    """
        input = self._permute(input)
        state_shape = [1, input.shape[1], self.hidden_size]
        state_shape = state_shape, state_shape
        h0, c0 = self._get_state(input, state, state_shape)
        h, c = self._impl(input, (h0[0], c0[0]), self._get_zoneout_mask(input))
        state = self._get_final_state((h, c), lengths)
        output = self._permute(h[1:])
        return output, state

    def _impl(self, input, state, zoneout_mask):
        return LayerNormLSTMScript(self.training, self.zoneout, input.contiguous(), state[0].contiguous(), state[1].contiguous(), self.kernel.contiguous(), F.dropout(self.recurrent_kernel, self.dropout, self.training).contiguous(), self.bias.contiguous(), self.gamma.contiguous(), self.gamma_h.contiguous(), self.beta_h.contiguous(), zoneout_mask.contiguous())


def LSTMScript(training: bool, zoneout_prob: float, input, h0, c0, kernel, recurrent_kernel, bias, zoneout_mask):
    time_steps = input.shape[0]
    batch_size = input.shape[1]
    hidden_size = recurrent_kernel.shape[0]
    h = [h0]
    c = [c0]
    Wx = input @ kernel
    for t in range(time_steps):
        v = h[t] @ recurrent_kernel + Wx[t] + bias
        i, g, f, o = torch.chunk(v, 4, 1)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        c.append(f * c[t] + i * g)
        h.append(o * torch.tanh(c[-1]))
        if zoneout_prob:
            if training:
                h[-1] = (h[-1] - h[-2]) * zoneout_mask[t] + h[-2]
            else:
                h[-1] = zoneout_prob * h[-2] + (1 - zoneout_prob) * h[-1]
    h = torch.stack(h)
    c = torch.stack(c)
    return h, c


class LSTM(BaseRNN):
    """
  Long Short-Term Memory layer.

  This LSTM layer offers a fused, GPU-accelerated PyTorch op for inference
  and training. Although this implementation is comparable in performance to
  cuDNN's LSTM, it offers additional options not typically found in other
  high-performance implementations. DropConnect and Zoneout regularization are
  built-in, and this layer allows setting a non-zero initial forget gate bias.

  See [\\_\\_init\\_\\_](#__init__) and [forward](#forward) for general usage.
  See [from_native_weights](#from_native_weights) and
  [to_native_weights](#to_native_weights) for compatibility with PyTorch LSTMs.
  """

    def __init__(self, input_size, hidden_size, batch_first=False, forget_bias=1.0, dropout=0.0, zoneout=0.0, return_state_sequence=False):
        """
    Initialize the parameters of the LSTM layer.

    Arguments:
      input_size: int, the feature dimension of the input.
      hidden_size: int, the feature dimension of the output.
      batch_first: (optional) bool, if `True`, then the input and output
        tensors are provided as `(batch, seq, feature)`.
      forget_bias: (optional) float, sets the initial bias of the forget gate
        for this LSTM cell.
      dropout: (optional) float, sets the dropout rate for DropConnect
        regularization on the recurrent matrix.
      zoneout: (optional) float, sets the zoneout rate for Zoneout
        regularization.
      return_state_sequence: (optional) bool, if `True`, the forward pass will
        return the entire state sequence instead of just the final state. Note
        that if the input is a padded sequence, the returned state will also
        be a padded sequence.

    Variables:
      kernel: the input projection weight matrix. Dimensions
        (input_size, hidden_size * 4) with `i,g,f,o` gate layout. Initialized
        with Xavier uniform initialization.
      recurrent_kernel: the recurrent projection weight matrix. Dimensions
        (hidden_size, hidden_size * 4) with `i,g,f,o` gate layout. Initialized
        with orthogonal initialization.
      bias: the projection bias vector. Dimensions (hidden_size * 4) with
        `i,g,f,o` gate layout. The forget gate biases are initialized to
        `forget_bias` and the rest are zeros.
    """
        super().__init__(input_size, hidden_size, batch_first, zoneout, return_state_sequence)
        if dropout < 0 or dropout > 1:
            raise ValueError('LSTM: dropout must be in [0.0, 1.0]')
        if zoneout < 0 or zoneout > 1:
            raise ValueError('LSTM: zoneout must be in [0.0, 1.0]')
        self.forget_bias = forget_bias
        self.dropout = dropout
        self.kernel = nn.Parameter(torch.empty(input_size, hidden_size * 4))
        self.recurrent_kernel = nn.Parameter(torch.empty(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.empty(hidden_size * 4))
        self.reset_parameters()

    def to_native_weights(self):
        """
    Converts Haste LSTM weights to native PyTorch LSTM weights.

    Returns:
      weight_ih_l0: Parameter, the input-hidden weights of the LSTM layer.
      weight_hh_l0: Parameter, the hidden-hidden weights of the LSTM layer.
      bias_ih_l0: Parameter, the input-hidden bias of the LSTM layer.
      bias_hh_l0: Parameter, the hidden-hidden bias of the LSTM layer.
    """

        def reorder_weights(w):
            i, g, f, o = torch.chunk(w, 4, dim=-1)
            return torch.cat([i, f, g, o], dim=-1)
        kernel = reorder_weights(self.kernel).permute(1, 0).contiguous()
        recurrent_kernel = reorder_weights(self.recurrent_kernel).permute(1, 0).contiguous()
        half_bias = reorder_weights(self.bias) / 2.0
        kernel = torch.nn.Parameter(kernel)
        recurrent_kernel = torch.nn.Parameter(recurrent_kernel)
        bias1 = torch.nn.Parameter(half_bias)
        bias2 = torch.nn.Parameter(half_bias.clone())
        return kernel, recurrent_kernel, bias1, bias2

    def from_native_weights(self, weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0):
        """
    Copies and converts the provided PyTorch LSTM weights into this layer.

    Arguments:
      weight_ih_l0: Parameter, the input-hidden weights of the PyTorch LSTM layer.
      weight_hh_l0: Parameter, the hidden-hidden weights of the PyTorch LSTM layer.
      bias_ih_l0: Parameter, the input-hidden bias of the PyTorch LSTM layer.
      bias_hh_l0: Parameter, the hidden-hidden bias of the PyTorch LSTM layer.
    """

        def reorder_weights(w):
            i, f, g, o = torch.chunk(w, 4, dim=-1)
            return torch.cat([i, g, f, o], dim=-1)
        kernel = reorder_weights(weight_ih_l0.permute(1, 0)).contiguous()
        recurrent_kernel = reorder_weights(weight_hh_l0.permute(1, 0)).contiguous()
        bias = reorder_weights(bias_ih_l0 + bias_hh_l0).contiguous()
        self.kernel = nn.Parameter(kernel)
        self.recurrent_kernel = nn.Parameter(recurrent_kernel)
        self.bias = nn.Parameter(bias)

    def reset_parameters(self):
        """Resets this layer's parameters to their initial values."""
        hidden_size = self.hidden_size
        for i in range(4):
            nn.init.xavier_uniform_(self.kernel[:, i * hidden_size:(i + 1) * hidden_size])
            nn.init.orthogonal_(self.recurrent_kernel[:, i * hidden_size:(i + 1) * hidden_size])
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.bias[hidden_size * 2:hidden_size * 3], self.forget_bias)

    def forward(self, input, state=None, lengths=None):
        """
    Runs a forward pass of the LSTM layer.

    Arguments:
      input: Tensor, a batch of input sequences to pass through the LSTM.
        Dimensions (seq_len, batch_size, input_size) if `batch_first` is
        `False`, otherwise (batch_size, seq_len, input_size).
      lengths: (optional) Tensor, list of sequence lengths for each batch
        element. Dimension (batch_size). This argument may be omitted if
        all batch elements are unpadded and have the same sequence length.

    Returns:
      output: Tensor, the output of the LSTM layer. Dimensions
        (seq_len, batch_size, hidden_size) if `batch_first` is `False` (default)
        or (batch_size, seq_len, hidden_size) if `batch_first` is `True`. Note
        that if `lengths` was specified, the `output` tensor will not be
        masked. It's the caller's responsibility to either not use the invalid
        entries or to mask them out before using them.
      (h_n, c_n): the hidden and cell states, respectively, for the last
        sequence item. Dimensions (1, batch_size, hidden_size).
    """
        input = self._permute(input)
        state_shape = [1, input.shape[1], self.hidden_size]
        state_shape = state_shape, state_shape
        h0, c0 = self._get_state(input, state, state_shape)
        h, c = self._impl(input, (h0[0], c0[0]), self._get_zoneout_mask(input))
        state = self._get_final_state((h, c), lengths)
        output = self._permute(h[1:])
        return output, state

    def _impl(self, input, state, zoneout_mask):
        return LSTMScript(self.training, self.zoneout, input.contiguous(), state[0].contiguous(), state[1].contiguous(), self.kernel.contiguous(), F.dropout(self.recurrent_kernel, self.dropout, self.training).contiguous(), self.bias.contiguous(), zoneout_mask.contiguous())


def NBRCScript(training: bool, zoneout_prob: float, input, h0, kernel, recurrent_kernel, bias, recurrent_bias, zoneout_mask):
    time_steps = input.shape[0]
    batch_size = input.shape[1]
    hidden_size = recurrent_kernel.shape[0]
    h = [h0]
    Wx = input @ kernel + bias
    for t in range(time_steps):
        Rh = h[t] @ recurrent_kernel + recurrent_bias
        vx = torch.chunk(Wx[t], 3, 1)
        vh = torch.chunk(Rh, 3, 1)
        z = torch.sigmoid(vx[0] + vh[0])
        r = torch.sigmoid(vx[1] + vh[1])
        g = torch.tanh(vx[2] + r * vh[2])
        h.append(z * h[t] + (1 - z) * g)
        if zoneout_prob:
            if training:
                h[-1] = (h[-1] - h[-2]) * zoneout_mask[t] + h[-2]
            else:
                h[-1] = zoneout_prob * h[-2] + (1 - zoneout_prob) * h[-1]
    h = torch.stack(h)
    return h


class NBRC(BaseRNN):
    """
  Gated Recurrent Unit layer.

  This NBRC layer offers a fused, GPU-accelerated PyTorch op for inference
  and training. There are two commonly-used variants of NBRC cells. This one
  implements 1406.1078v1 which applies the reset gate to the hidden state
  after matrix multiplication. cuDNN also implements this variant. The other
  variant, 1406.1078v3, applies the reset gate before matrix multiplication
  and is currently unsupported.

  This layer has built-in support for DropConnect and Zoneout, which are
  both techniques used to regularize RNNs.

  See [\\_\\_init\\_\\_](#__init__) and [forward](#forward) for usage.
  See [from_native_weights](#from_native_weights) and
  [to_native_weights](#to_native_weights) for compatibility with PyTorch NBRCs.
  """

    def __init__(self, input_size, hidden_size, batch_first=False, dropout=0.0, zoneout=0.0, return_state_sequence=False):
        """
    Initialize the parameters of the NBRC layer.

    Arguments:
      input_size: int, the feature dimension of the input.
      hidden_size: int, the feature dimension of the output.
      batch_first: (optional) bool, if `True`, then the input and output
        tensors are provided as `(batch, seq, feature)`.
      dropout: (optional) float, sets the dropout rate for DropConnect
        regularization on the recurrent matrix.
      zoneout: (optional) float, sets the zoneout rate for Zoneout
        regularization.
      return_state_sequence: (optional) bool, if `True`, the forward pass will
        return the entire state sequence instead of just the final state. Note
        that if the input is a padded sequence, the returned state will also
        be a padded sequence.

    Variables:
      kernel: the input projection weight matrix. Dimensions
        (input_size, hidden_size * 3) with `z,r,h` gate layout. Initialized
        with Xavier uniform initialization.
      recurrent_kernel: the recurrent projection weight matrix. Dimensions
        (hidden_size, hidden_size * 3) with `z,r,h` gate layout. Initialized
        with orthogonal initialization.
      bias: the input projection bias vector. Dimensions (hidden_size * 3) with
        `z,r,h` gate layout. Initialized to zeros.
      recurrent_bias: the recurrent projection bias vector. Dimensions
        (hidden_size * 3) with `z,r,h` gate layout. Initialized to zeros.
    """
        super().__init__(input_size, hidden_size, batch_first, zoneout, return_state_sequence)
        if dropout < 0 or dropout > 1:
            raise ValueError('NBRC: dropout must be in [0.0, 1.0]')
        if zoneout < 0 or zoneout > 1:
            raise ValueError('NBRC: zoneout must be in [0.0, 1.0]')
        self.dropout = dropout
        self.kernel = nn.Parameter(torch.empty(input_size, hidden_size * 3))
        self.recurrent_kernel = nn.Parameter(torch.empty(hidden_size, hidden_size * 3))
        self.bias = nn.Parameter(torch.empty(hidden_size * 3))
        self.recurrent_bias = nn.Parameter(torch.empty(hidden_size * 3))
        self.reset_parameters()

    def to_native_weights(self):
        """
    Converts Haste NBRC weights to native PyTorch NBRC weights.

    Returns:
      weight_ih_l0: Parameter, the input-hidden weights of the NBRC layer.
      weight_hh_l0: Parameter, the hidden-hidden weights of the NBRC layer.
      bias_ih_l0: Parameter, the input-hidden bias of the NBRC layer.
      bias_hh_l0: Parameter, the hidden-hidden bias of the NBRC layer.
    """

        def reorder_weights(w):
            z, r, n = torch.chunk(w, 3, dim=-1)
            return torch.cat([r, z, n], dim=-1)
        kernel = reorder_weights(self.kernel).permute(1, 0).contiguous()
        recurrent_kernel = reorder_weights(self.recurrent_kernel).permute(1, 0).contiguous()
        bias1 = reorder_weights(self.bias).contiguous()
        bias2 = reorder_weights(self.recurrent_bias).contiguous()
        kernel = torch.nn.Parameter(kernel)
        recurrent_kernel = torch.nn.Parameter(recurrent_kernel)
        bias1 = torch.nn.Parameter(bias1)
        bias2 = torch.nn.Parameter(bias2)
        return kernel, recurrent_kernel, bias1, bias2

    def from_native_weights(self, weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0):
        """
    Copies and converts the provided PyTorch NBRC weights into this layer.

    Arguments:
      weight_ih_l0: Parameter, the input-hidden weights of the PyTorch NBRC layer.
      weight_hh_l0: Parameter, the hidden-hidden weights of the PyTorch NBRC layer.
      bias_ih_l0: Parameter, the input-hidden bias of the PyTorch NBRC layer.
      bias_hh_l0: Parameter, the hidden-hidden bias of the PyTorch NBRC layer.
    """

        def reorder_weights(w):
            r, z, n = torch.chunk(w, 3, axis=-1)
            return torch.cat([z, r, n], dim=-1)
        kernel = reorder_weights(weight_ih_l0.permute(1, 0)).contiguous()
        recurrent_kernel = reorder_weights(weight_hh_l0.permute(1, 0)).contiguous()
        bias = reorder_weights(bias_ih_l0).contiguous()
        recurrent_bias = reorder_weights(bias_hh_l0).contiguous()
        self.kernel = nn.Parameter(kernel)
        self.recurrent_kernel = nn.Parameter(recurrent_kernel)
        self.bias = nn.Parameter(bias)
        self.recurrent_bias = nn.Parameter(recurrent_bias)

    def reset_parameters(self):
        """Resets this layer's parameters to their initial values."""
        hidden_size = self.hidden_size
        for i in range(3):
            nn.init.xavier_uniform_(self.kernel[:, i * hidden_size:(i + 1) * hidden_size])
            nn.init.orthogonal_(self.recurrent_kernel[:, i * hidden_size:(i + 1) * hidden_size])
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.recurrent_bias)

    def forward(self, input, state=None, lengths=None):
        """
    Runs a forward pass of the NBRC layer.

    Arguments:
      input: Tensor, a batch of input sequences to pass through the NBRC.
        Dimensions (seq_len, batch_size, input_size) if `batch_first` is
        `False`, otherwise (batch_size, seq_len, input_size).
      lengths: (optional) Tensor, list of sequence lengths for each batch
        element. Dimension (batch_size). This argument may be omitted if
        all batch elements are unpadded and have the same sequence length.

    Returns:
      output: Tensor, the output of the NBRC layer. Dimensions
        (seq_len, batch_size, hidden_size) if `batch_first` is `False` (default)
        or (batch_size, seq_len, hidden_size) if `batch_first` is `True`. Note
        that if `lengths` was specified, the `output` tensor will not be
        masked. It's the caller's responsibility to either not use the invalid
        entries or to mask them out before using them.
      h_n: the hidden state for the last sequence item. Dimensions
        (1, batch_size, hidden_size).
    """
        input = self._permute(input)
        state_shape = [1, input.shape[1], self.hidden_size]
        h0 = self._get_state(input, state, state_shape)
        h = self._impl(input, h0[0], self._get_zoneout_mask(input))
        state = self._get_final_state(h, lengths)
        output = self._permute(h[1:])
        return output, state

    def _impl(self, input, state, zoneout_mask):
        return NBRCScript(self.training, self.zoneout, input.contiguous(), state.contiguous(), self.kernel.contiguous(), F.dropout(self.recurrent_kernel, self.dropout, self.training).contiguous(), self.bias.contiguous(), self.recurrent_bias.contiguous(), zoneout_mask.contiguous())


def _mish_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


def _mish_fwd(x):
    return x.mul(torch.tanh(F.softplus(x)))


class MishAutoFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _mish_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        return _mish_bwd(x, grad_output)


class _Mish(nn.Module):

    def forward(self, x):
        return MishAutoFn.apply(x)


class LM(nn.Module):

    def __init__(self, vocab_sz, embed_sz, hidden_sz, num_layers, p=0.2, **kwargs):
        super(LM, self).__init__()
        self.embed = nn.Embedding(vocab_sz, embed_sz, padding_idx=0)
        self.rnn = nn.LSTM(embed_sz, hidden_sz, batch_first=True, num_layers=num_layers)
        self.drop = nn.Dropout(p)
        self.linear = nn.Linear(hidden_sz, vocab_sz)
        if embed_sz == hidden_sz:
            self.linear.weight = self.embed.weight

    def forward(self, x, state=None):
        x = self.embed(x)
        if state:
            x, state = self.rnn(x, state)
        else:
            x, state = self.rnn(x)
        x = self.drop(x)
        x = self.linear(x)
        x = F.log_softmax(x, dim=-1)
        return x, state


class ResidualAdapter(Module):
    """
    ResidualAdapter according to
    https://ai.googleblog.com/2019/09/large-scale-multilingual-speech.html?m=1
    """

    def __init__(self, hidden_sz, projection='fcnet', projection_factor=3.2, activation=F.relu):
        self.hidden_sz = hidden_sz
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(hidden_sz)
        if projection == 'conv':
            pass
        else:
            bottleneck_sz = int(hidden_sz / projection_factor)
            bottleneck_sz = bottleneck_sz + (8 - bottleneck_sz % 8)
            self.down = nn.Linear(hidden_sz, bottleneck_sz)
            self.up = nn.Linear(bottleneck_sz, hidden_sz)

    def forward(self, x):
        inp = x
        x = self.layer_norm(x)
        x = self.down(x)
        x = self.activation(x)
        x = self.up(x)
        return x + inp


class Encoder(Module):

    def __init__(self, feature_sz, hidden_sz, out_sz, dropout=0.01, num_layers=2, trace=True, device='cuda:0', layer_norm=False, rnn_type='LSTM', use_tmp_state_pcent=0.9, **kwargs):
        self.num_layers = num_layers
        self.input_norm = nn.LayerNorm(feature_sz)
        self.rnn_stack = CustomCPURNN(feature_sz, hidden_sz, num_layers, rnn_type=rnn_type, reduction_indices=[], reduction_factors=[], layer_norm=layer_norm, rezero=False, utsp=use_tmp_state_pcent)
        self.drop = nn.Dropout(dropout)
        if not hidden_sz == out_sz:
            self.linear = nn.Linear(hidden_sz, out_sz)
        else:
            self.linear = nn.Sequential()

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x, state=None, lengths=None, return_state=False):
        x = x.reshape((x.size(0), x.size(1), -1))
        x = self.input_norm(x)
        x, state = self.rnn_stack(x, state=state, lengths=lengths)
        x = self.drop(x)
        x = self.linear(x)
        if return_state:
            return x, state
        return x


class Joint(Module):

    def __init__(self, out_sz, joint_sz, vocab_sz, joint_method):
        self.joint_method = joint_method
        if joint_method == 'add':
            input_sz = out_sz
        elif joint_method == 'concat':
            input_sz = 2 * out_sz
        else:
            raise Exception('No such joint_method')
        self.joint = nn.Sequential(nn.Linear(input_sz, joint_sz), nn.Tanh(), nn.Linear(joint_sz, vocab_sz))

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, h_pred, h_enc):
        if self.joint_method == 'add':
            x = h_pred + h_enc
        elif self.joint_method == 'concat':
            x = torch.cat((h_pred, h_enc), dim=-1)
        else:
            raise Exception('No such joint_method')
        x = self.joint(x)
        return x


class Predictor(Module):

    def __init__(self, vocab_sz, embed_sz, hidden_sz, out_sz, dropout=0.01, num_layers=2, blank=0, layer_norm=False, rnn_type='NBRC', use_tmp_state_pcent=0.9):
        self.vocab_sz = vocab_sz
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_sz, embed_sz, padding_idx=blank)
        if not embed_sz == hidden_sz:
            self.ffn = nn.Linear(embed_sz, hidden_sz)
        else:
            self.ffn = nn.Sequential()
        self.rnn_stack = CustomCPURNN(hidden_sz, hidden_sz, num_layers, rnn_type=rnn_type, layer_norm=layer_norm, utsp=use_tmp_state_pcent)
        self.drop = nn.Dropout(dropout)
        if not hidden_sz == out_sz:
            self.linear = nn.Linear(hidden_sz, out_sz)
        else:
            self.linear = nn.Sequential()

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x, state=None, lengths=None):
        x = self.embed(x)
        x = self.ffn(x)
        x, state = self.rnn_stack(x, state=state, lengths=lengths)
        x = self.drop(x)
        x = self.linear(x)
        return x, state


ALPHA = 0.1


DEBUG = False


MIN_VAL = -10.0


THETA = 1.0


def standardize(t, eps=1e-05):
    t.add_(-t.mean())
    t.div_(t.std() + eps)


class LMFuser:

    def __init__(self, lm):
        self.lm = lm
        self.lm_logits = None
        self.lm_state = None
        self.has_lm = self.lm is not None

    def advance(self, y_one_char):
        if self.has_lm:
            self.lm_logits, self.lm_state = self.lm(y_one_char, self.lm_state)
            standardize(self.lm_logits)
            self.lm_logits[:, :, 0] = MIN_VAL

    def fuse(self, joint_out, prob, pred, alpha=ALPHA, theta=THETA):
        lm_logits = self.lm_logits
        if self.has_lm and torch.is_tensor(lm_logits):
            standardize(joint_out)
            joint_out[:, :, :, 0] = MIN_VAL
            if DEBUG:
                None
                None
            fused = alpha * lm_logits + theta * joint_out
            prob, pred = fused.max(-1)
            return fused, prob, pred
        return joint_out, prob, pred

    def reset(self):
        self.lm_logits = None
        self.lm_state = None


class Transducer(Module):

    def __init__(self, feature_sz, embed_sz, vocab_sz, hidden_sz, out_sz, joint_sz, lang, l_e=6, l_p=2, p_j=0.0, blank=0, joint_method='concat', perf=False, act=F.relu, use_tmp_bos=True, use_tmp_bos_pcent=0.99, encoder_kwargs={}, predictor_kwargs={}, **kwargs):
        self.encoder = Encoder(feature_sz, hidden_sz=hidden_sz, out_sz=out_sz, **encoder_kwargs)
        self.predictor = Predictor(vocab_sz, embed_sz=embed_sz, hidden_sz=hidden_sz, out_sz=out_sz, **predictor_kwargs)
        self.joint = Joint(out_sz, joint_sz, vocab_sz, joint_method)
        self.lang = lang
        self.blank = blank
        self.bos = 2
        self.perf = perf
        self.mp = False
        self.bos_cache = {}
        self.use_tmp_bos = use_tmp_bos
        self.use_tmp_bos_pcent = use_tmp_bos_pcent
        self.vocab_sz = vocab_sz
        self.lm = None

    @staticmethod
    def from_config(conf, lang, lm=None):
        m = Transducer(conf['model']['feature_sz'], conf['model']['embed_sz'], conf['model']['vocab_sz'], conf['model']['hidden_sz'], conf['model']['out_sz'], conf['model']['joint_sz'], lang, p_e=conf['model']['encoder']['dropout'], p_p=conf['model']['predictor']['dropout'], p_j=conf['model']['joint']['dropout'], joint_method=conf['model']['joint']['method'], perf=False, bs=conf['bs'], raw_audio=False, use_tmp_bos=conf['model']['use_tmp_bos'], use_tmp_bos_pcent=conf['model']['use_tmp_bos_pcent'], encoder_kwargs=conf['model']['encoder'], predictor_kwargs=conf['model']['predictor'])
        m.mp = conf['mp']
        return m

    def param_groups(self):
        return [self.encoder.param_groups(), self.predictor.param_groups(), self.joint.param_groups()]

    def convert_to_cpu(self):
        self.encoder.rnn_stack = self.encoder.rnn_stack.convert_to_cpu()
        self.predictor.rnn_stack = self.predictor.rnn_stack.convert_to_cpu()
        return self

    def convert_to_gpu(self):
        self.encoder.rnn_stack = self.encoder.rnn_stack.convert_to_gpu()
        self.predictor.rnn_stack = self.predictor.rnn_stack.convert_to_gpu()
        return self

    def start_perf(self):
        if self.perf:
            self.t = time.time()

    def stop_perf(self, name='unknown'):
        if self.perf:
            t = (time.time() - self.t) * 1000.0
            None

    def grab_bos(self, y, yl, bs, device):
        if self.training and self.use_tmp_bos:
            r = random.random()
            thresh = 1.0 - self.use_tmp_bos_pcent
            cached_bos = self.bos_cache.get(bs)
            if torch.is_tensor(cached_bos) and r > thresh:
                bos = cached_bos
                return bos
            try:
                q = torch.clamp(yl[:, None] - 1, min=0)
                self.bos_cache[bs] = y.gather(1, q).detach()
            except:
                pass
        bos = torch.zeros((bs, 1), device=device).long()
        bos = bos.fill_(self.bos)
        return bos

    def forward(self, tpl):
        """
        (x, y)
        x: N tuples (audios of shape [N, n_chans, seq_len, H], x_lens)
        y: N tuples (y_padded, y_lens)
        """
        x, y, xl, yl = tpl
        if self.mp:
            x = x.half()
        self.start_perf()
        x = x.reshape(x.size(0), x.size(1), -1)
        encoder_out = self.encoder(x, lengths=xl)
        self.stop_perf('encoder')
        N, T, H = encoder_out.size()
        bos = self.grab_bos(y, yl, bs=N, device=encoder_out.device)
        yconcat = torch.cat((bos, y), dim=1)
        self.start_perf()
        predictor_out, _ = self.predictor(yconcat, lengths=yl)
        self.stop_perf('predictor')
        U = predictor_out.size(1)
        M = max(T, U)
        sz = N, T, U, H
        encoder_out = encoder_out.unsqueeze(2).expand(sz).contiguous()
        predictor_out = predictor_out.unsqueeze(1).expand(sz).contiguous()
        self.start_perf()
        joint_out = self.joint(predictor_out, encoder_out)
        self.stop_perf('joint')
        joint_out = F.log_softmax(joint_out, -1)
        return joint_out

    def decode(self, *args, **kwargs):
        res, log_p, _ = self.decode_greedy(*args, **kwargs)
        return res, log_p

    def transcribe(self, *args, **kwargs):
        res, _, metrics, _ = self.decode_greedy(*args, **kwargs)
        return res, metrics

    def decode_greedy(self, x, max_iters=3, alpha=0.005, theta=1.0):
        """x must be of shape [C, T, H]"""
        metrics = {}
        extra = {'iters': [], 'outs': []}
        self.eval()
        self.encoder.eval()
        self.predictor.eval()
        self.joint.eval()
        if len(x.shape) == 2:
            x = x[None]
        x = x[None]
        encoder_out = self.encoder(x)[0]
        y_one_char = torch.LongTensor([[self.bos]])
        h_t_pred, pred_state = self.predictor(y_one_char)
        fuser = LMFuser(self.lm)
        y_seq, log_p = [], 0.0
        for h_t_enc in encoder_out:
            iters = 0
            while iters < max_iters:
                iters += 1
                _h_t_pred = h_t_pred[None]
                _h_t_enc = h_t_enc[None, None, None]
                joint_out = self.joint(_h_t_pred, _h_t_enc)
                joint_out = F.log_softmax(joint_out, dim=-1)
                extra['outs'].append(joint_out.clone())
                prob, pred = joint_out.max(-1)
                pred = int(pred)
                log_p += float(prob)
                if pred == self.blank:
                    break
                else:
                    joint_out, prob, pred = fuser.fuse(joint_out, prob, pred)
                    y_seq.append(pred)
                    y_one_char[0][0] = pred
                    h_t_pred, pred_state = self.predictor(y_one_char, state=pred_state)
                    fuser.advance(y_one_char)
            extra['iters'].append(iters)
        align = np.array(extra['iters'])
        _sum = align.sum()
        val, cnt = np.unique(align, return_counts=True)
        d = {v: c for v, c in zip(val, cnt)}
        _ones = d.get(1, 0)
        alignment_score = (_sum - _ones) / (_sum + 0.0001)
        metrics['alignment_score'] = alignment_score
        return self.lang.denumericalize(y_seq), -log_p, metrics, extra

    def transcribe_stream(self, stream, denumericalizer, max_iters=10, alpha=0.3, theta=1.0):
        """
        stream is expected to yield chunks of shape (NCHANS, CHUNKSIZE)
        """
        self.eval()
        encoder_state = None
        predictor_state = None
        y_one_char = torch.LongTensor([[self.bos]])
        h_t_pred = None
        y = []
        fuser = LMFuser(self.lm)

        def reset_encoder():
            nonlocal encoder_state
            encoder_state = None

        def reset_predictor():
            nonlocal y_one_char, h_t_pred, predictor_state
            y_one_char = torch.LongTensor([[self.bos]])
            h_t_pred, predictor_state = self.predictor(y_one_char)

        def reset_lm():
            fuser.reset()

        def reset():
            reset_encoder()
            reset_predictor()
            reset_lm()
        reset()
        blanks = 0
        nonblanks = 0
        for chunk in stream:
            if chunk is None:
                continue
            chunk = chunk[None]
            self.start_perf()
            if encoder_state is None:
                encoder_out, encoder_state = self.encoder(chunk, return_state=True)
            else:
                encoder_out, encoder_state = self.encoder(chunk, state=encoder_state, return_state=True)
            h_t_enc = encoder_out[0]
            self.stop_perf('encoder')
            self.start_perf()
            y_seq = []
            for i in range(h_t_enc.size(-2)):
                h_enc = h_t_enc[..., i, :]
                iters = 0
                while iters < max_iters:
                    iters += 1
                    _h_t_pred = h_t_pred[None]
                    _h_t_enc = h_enc[None, None, None]
                    joint_out = self.joint(_h_t_pred, _h_t_enc)
                    joint_out = F.log_softmax(joint_out, dim=-1)
                    prob, pred = joint_out.max(-1)
                    pred = int(pred)
                    if pred == self.blank:
                        blanks += 1
                        break
                    else:
                        joint_out, prob, pred = fuser.fuse(joint_out, prob, pred)
                        y_seq.append(pred)
                        y_one_char[0][0] = pred
                        h_t_pred, predictor_state = self.predictor(y_one_char, state=predictor_state)
                        fuser.advance(y_one_char)
                        nonblanks += 1
            y = y + y_seq
            yield y, denumericalizer(y_seq), reset
            self.stop_perf('joint + predictor')


class CTCModel(Module):

    def __init__(self):
        layer = nn.TransformerEncoderLayer(128, 8)
        self.encoder = nn.TransformerEncoder(layer, 8)
        self.linear = nn.Linear(128, 2048)

    def convert_to_gpu(self):
        pass

    def param_groups(self):
        return [p for p in self.parameters() if p.requires_grad]

    @staticmethod
    def from_config(conf, lang):
        return CTCModel()

    def forward(self, tpl):
        x, y, xl, yl = tpl
        x = x.view(x.size(1), x.size(0), -1).contiguous()
        x = self.encoder(x)
        x = self.linear(x)
        x = F.log_softmax(x, -1)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GRU,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LSTM,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (LayerNormLSTM,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (NBRC,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (_Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_iceychris_LibreASR(_paritybench_base):
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

