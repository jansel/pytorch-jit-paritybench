import sys
_module = sys.modules[__name__]
del sys
ner = _module
parsing = _module
pos_tagging = _module
neuronlp2 = _module
io = _module
alphabet = _module
common = _module
conll03_data = _module
conllx_data = _module
conllx_stacked_data = _module
instance = _module
logger = _module
reader = _module
utils = _module
writer = _module
models = _module
parsing = _module
sequence_labeling = _module
nn = _module
_functions = _module
rnnFusedBackend = _module
skipconnect_rnn = _module
variational_rnn = _module
crf = _module
init = _module
modules = _module
skip_rnn = _module
utils = _module
variational_rnn = _module
optim = _module
lr_scheduler = _module
tasks = _module
parser = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import time


import numpy as np


import torch


from torch.optim.adamw import AdamW


from torch.optim import SGD


from torch.nn.utils import clip_grad_norm_


import math


from collections import defaultdict


from collections import OrderedDict


from enum import Enum


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import pack_padded_sequence


from torch.autograd.function import Function


from torch.autograd.function import once_differentiable


from torch.nn import functional as F


from torch.nn.parameter import Parameter


import collections


from itertools import repeat


from torch._six import inf


from torch.optim.optimizer import Optimizer


class BiLinear(nn.Module):
    """
    Bi-linear layer
    """

    def __init__(self, left_features, right_features, out_features, bias=True):
        """

        Args:
            left_features: size of left input
            right_features: size of right input
            out_features: size of output
            bias: If set to False, the layer will not learn an additive bias.
                Default: True
        """
        super(BiLinear, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features
        self.U = Parameter(torch.Tensor(self.out_features, self.left_features, self.right_features))
        self.weight_left = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.weight_right = Parameter(torch.Tensor(self.out_features, self.right_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_left)
        nn.init.xavier_uniform_(self.weight_right)
        nn.init.constant_(self.bias, 0.0)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left, input_right):
        """

        Args:
            input_left: Tensor
                the left input tensor with shape = [batch1, batch2, ..., left_features]
            input_right: Tensor
                the right input tensor with shape = [batch1, batch2, ..., right_features]

        Returns:

        """
        batch_size = input_left.size()[:-1]
        batch = int(np.prod(batch_size))
        input_left = input_left.view(batch, self.left_features)
        input_right = input_right.view(batch, self.right_features)
        output = F.bilinear(input_left, input_right, self.U, self.bias)
        output = output + F.linear(input_left, self.weight_left, None) + F.linear(input_right, self.weight_right, None)
        return output.view(batch_size + (self.out_features,))

    def __repr__(self):
        return self.__class__.__name__ + ' (' + 'left_features=' + str(self.left_features) + ', right_features=' + str(self.right_features) + ', out_features=' + str(self.out_features) + ')'


class CharCNN(nn.Module):
    """
    CNN layers for characters
    """

    def __init__(self, num_layers, in_channels, out_channels, hidden_channels=None, activation='elu'):
        super(CharCNN, self).__init__()
        assert activation in ['elu', 'tanh']
        if activation == 'elu':
            ACT = nn.ELU
        else:
            ACT = nn.Tanh
        layers = list()
        for i in range(num_layers - 1):
            layers.append(('conv{}'.format(i), nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)))
            layers.append(('act{}'.format(i), ACT()))
            in_channels = hidden_channels
        layers.append(('conv_top', nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)))
        layers.append(('act_top', ACT()))
        self.act = ACT
        self.net = nn.Sequential(OrderedDict(layers))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv1d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
            else:
                assert isinstance(layer, self.act)

    def forward(self, char):
        """

        Args:
            char: Tensor
                the input tensor of character [batch, sent_length, char_length, in_channels]

        Returns: Tensor
            output character encoding with shape [batch, sent_length, in_channels]

        """
        char_size = char.size()
        char = char.view(-1, char_size[2], char_size[3]).transpose(1, 2)
        char = self.net(char).max(dim=2)[0]
        return char.view(char_size[0], char_size[1], -1)


class VarRNNCellBase(nn.Module):

    def __repr__(self):
        s = '{name}({model_dim}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != 'tanh':
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def reset_noise(self, batch_size):
        """
        Should be overriden by all subclasses.
        Args:
            batch_size: (int) batch size of input.
        """
        raise NotImplementedError


class VarFastLSTMCell(VarRNNCellBase):
    """
    A long short-term memory (LSTM) cell with variational dropout.

    .. math::

        egin{array}{ll}
        i = \\mathrm{sigmoid}(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \\mathrm{sigmoid}(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = 	anh(W_{ig} x + b_{ig} + W_{hc} h + b_{hg}) \\
        o = \\mathrm{sigmoid}(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * 	anh(c') \\
        \\end{array}

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: True
        p: (p_in, p_hidden) (tuple, optional): the drop probability for input and hidden. Default: (0.5, 0.5)

    Inputs: input, (h_0, c_0)
        - **input** (batch, model_dim): tensor containing input features
        - **h_0** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.
        - **c_0** (batch. hidden_size): tensor containing the initial cell state
          for each element in the batch.

    Outputs: h_1, c_1
        - **h_1** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch
        - **c_1** (batch, hidden_size): tensor containing the next cell state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(4*hidden_size x model_dim)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(4*hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`
    """

    def __init__(self, input_size, hidden_size, bias=True, p=(0.5, 0.5)):
        super(VarFastLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        p_in, p_hidden = p
        if p_in < 0 or p_in > 1:
            raise ValueError('input dropout probability has to be between 0 and 1, but got {}'.format(p_in))
        if p_hidden < 0 or p_hidden > 1:
            raise ValueError('hidden state dropout probability has to be between 0 and 1, but got {}'.format(p_hidden))
        self.p_in = p_in
        self.p_hidden = p_hidden
        self.noise_in = None
        self.noise_hidden = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.xavier_uniform_(self.weight_ih)
        if self.bias:
            nn.init.constant_(self.bias_hh, 0.0)
            nn.init.constant_(self.bias_ih, 0.0)

    def reset_noise(self, batch_size):
        if self.training:
            if self.p_in:
                noise = self.weight_ih.new_empty(batch_size, self.input_size)
                self.noise_in = noise.bernoulli_(1.0 - self.p_in) / (1.0 - self.p_in)
            else:
                self.noise_in = None
            if self.p_hidden:
                noise = self.weight_hh.new_empty(batch_size, self.hidden_size)
                self.noise_hidden = noise.bernoulli_(1.0 - self.p_hidden) / (1.0 - self.p_hidden)
            else:
                self.noise_hidden = None
        else:
            self.noise_in = None
            self.noise_hidden = None

    def forward(self, input, hx):
        return rnn_F.VarFastLSTMCell(input, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, self.noise_in, self.noise_hidden)


class VarRNNBase(nn.Module):

    def __init__(self, Cell, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=(0, 0), bidirectional=False, **kwargs):
        super(VarRNNBase, self).__init__()
        self.Cell = Cell
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.lstm = False
        num_directions = 2 if bidirectional else 1
        self.all_cells = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                cell = self.Cell(layer_input_size, hidden_size, self.bias, p=dropout, **kwargs)
                self.all_cells.append(cell)
                self.add_module('cell%d' % (layer * num_directions + direction), cell)

    def reset_parameters(self):
        for cell in self.all_cells:
            cell.reset_parameters()

    def reset_noise(self, batch_size):
        for cell in self.all_cells:
            cell.reset_noise(batch_size)

    def forward(self, input, mask=None, hx=None):
        batch_size = input.size(0) if self.batch_first else input.size(1)
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size)
            if self.lstm:
                hx = hx, hx
        func = rnn_F.AutogradVarRNN(num_layers=self.num_layers, batch_first=self.batch_first, bidirectional=self.bidirectional, lstm=self.lstm)
        self.reset_noise(batch_size)
        output, hidden = func(input, self.all_cells, hx, None if mask is None else mask.view(mask.size() + (1,)))
        return output, hidden

    def step(self, input, hx=None, mask=None):
        """
        execute one step forward (only for one-directional RNN).
        Args:
            input (batch, model_dim): input tensor of this step.
            hx (num_layers, batch, hidden_size): the hidden state of last step.
            mask (batch): the mask tensor of this step.

        Returns:
            output (batch, hidden_size): tensor containing the output of this step from the last layer of RNN.
            hn (num_layers, batch, hidden_size): tensor containing the hidden state of this step
        """
        assert not self.bidirectional, 'step only cannot be applied to bidirectional RNN.'
        batch_size = input.size(0)
        if hx is None:
            hx = input.new_zeros(self.num_layers, batch_size, self.hidden_size)
            if self.lstm:
                hx = hx, hx
        func = rnn_F.AutogradVarRNNStep(num_layers=self.num_layers, lstm=self.lstm)
        output, hidden = func(input, self.all_cells, hx, mask)
        return output, hidden


class VarFastLSTM(VarRNNBase):
    """Applies a multi-layer long short-term memory (LSTM) RNN to an input
    sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

            \\begin{array}{ll}
            i_t = \\mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\\\
            f_t = \\mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\\\
            g_t = \\tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\\\
            o_t = \\mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\\\
            c_t = f_t * c_{(t-1)} + i_t * g_t \\\\
            h_t = o_t * \\tanh(c_t)
            \\end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`x_t` is the hidden state of the previous layer at
    time `t` or :math:`input_t` for the first layer, and :math:`i_t`,
    :math:`f_t`, :math:`g_t`, :math:`o_t` are the input, forget, cell,
    and out gates, respectively.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        bias: If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: (dropout_in, dropout_hidden) tuple.
            If non-zero, introduces a dropout layer on the input and hidden of the each
            RNN layer with dropout rate dropout_in and dropout_hidden, resp.
        bidirectional: If True, becomes a bidirectional RNN. Default: False

    Inputs: input, mask, (h_0, c_0)
        - **input** (seq_len, batch, model_dim): tensor containing the features
          of the input sequence.
          **mask** (seq_len, batch): 0-1 tensor containing the mask of the input sequence.
        - **h_0** (num_layers \\* num_directions, batch, hidden_size): tensor
          containing the initial hidden state for each element in the batch.
        - **c_0** (num_layers \\* num_directions, batch, hidden_size): tensor
          containing the initial cell state for each element in the batch.


    Outputs: output, (h_n, c_n)
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for t=seq_len
        - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the cell state for t=seq_len
    """

    def __init__(self, *args, **kwargs):
        super(VarFastLSTM, self).__init__(VarFastLSTMCell, *args, **kwargs)
        self.lstm = True


class VarGRUCell(VarRNNCellBase):
    """A gated recurrent unit (GRU) cell with variational dropout.

    .. math::

        egin{array}{ll}
        r = \\mathrm{sigmoid}(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \\mathrm{sigmoid}(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = 	anh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \\end{array}

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: `True`
        p: (p_in, p_hidden) (tuple, optional): the drop probability for input and hidden. Default: (0.5, 0.5)

    Inputs: input, hidden
        - **input** (batch, model_dim): tensor containing input features
        - **hidden** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.

    Outputs: h'
        - **h'**: (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(3 x model_dim x hidden_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(3x hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(3 x hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(3 x hidden_size)`
    """

    def __init__(self, input_size, hidden_size, bias=True, p=(0.5, 0.5)):
        super(VarGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3, input_size, hidden_size))
        self.weight_hh = Parameter(torch.Tensor(3, hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3, hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3, hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        p_in, p_hidden = p
        if p_in < 0 or p_in > 1:
            raise ValueError('input dropout probability has to be between 0 and 1, but got {}'.format(p_in))
        if p_hidden < 0 or p_hidden > 1:
            raise ValueError('hidden state dropout probability has to be between 0 and 1, but got {}'.format(p_hidden))
        self.p_in = p_in
        self.p_hidden = p_hidden
        self.noise_in = None
        self.noise_hidden = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.xavier_uniform_(self.weight_ih)
        if self.bias:
            nn.init.constant_(self.bias_hh, 0.0)
            nn.init.constant_(self.bias_ih, 0.0)

    def reset_noise(self, batch_size):
        if self.training:
            if self.p_in:
                noise = self.weight_ih.new_empty(3, batch_size, self.input_size)
                self.noise_in = noise.bernoulli_(1.0 - self.p_in) / (1.0 - self.p_in)
            else:
                self.noise_in = None
            if self.p_hidden:
                noise = self.weight_hh.new_empty(3, batch_size, self.hidden_size)
                self.noise_hidden = noise.bernoulli_(1.0 - self.p_hidden) / (1.0 - self.p_hidden)
            else:
                self.noise_hidden = None
        else:
            self.noise_in = None
            self.noise_hidden = None

    def forward(self, input, hx):
        return rnn_F.VarGRUCell(input, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, self.noise_in, self.noise_hidden)


class VarGRU(VarRNNBase):
    """Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

            \\begin{array}{ll}
            r_t = \\mathrm{sigmoid}(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\\\
            z_t = \\mathrm{sigmoid}(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\\\
            n_t = \\tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\\\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)} \\\\
            \\end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the hidden
    state of the previous layer at time `t` or :math:`input_t` for the first
    layer, and :math:`r_t`, :math:`z_t`, :math:`n_t` are the reset, input,
    and new gates, respectively.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        nonlinearity: The non-linearity to use ['tanh'|'relu']. Default: 'tanh'
        bias: If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: (dropout_in, dropout_hidden) tuple.
            If non-zero, introduces a dropout layer on the input and hidden of the each
            RNN layer with dropout rate dropout_in and dropout_hidden, resp.
        bidirectional: If True, becomes a bidirectional RNN. Default: False

    Inputs: input, mask, h_0
        - **input** (seq_len, batch, model_dim): tensor containing the features
          of the input sequence.
          **mask** (seq_len, batch): 0-1 tensor containing the mask of the input sequence.
        - **h_0** (num_layers * num_directions, batch, hidden_size): tensor
          containing the initial hidden state for each element in the batch.

    Outputs: output, h_n
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features (h_k) from the last layer of the RNN,
          for each k.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
          been given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for k=seq_len.
    """

    def __init__(self, *args, **kwargs):
        super(VarGRU, self).__init__(VarGRUCell, *args, **kwargs)


class VarLSTMCell(VarRNNCellBase):
    """
    A long short-term memory (LSTM) cell with variational dropout.

    .. math::

        egin{array}{ll}
        i = \\mathrm{sigmoid}(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \\mathrm{sigmoid}(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = 	anh(W_{ig} x + b_{ig} + W_{hc} h + b_{hg}) \\
        o = \\mathrm{sigmoid}(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * 	anh(c') \\
        \\end{array}

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: True
        p: (p_in, p_hidden) (tuple, optional): the drop probability for input and hidden. Default: (0.5, 0.5)

    Inputs: input, (h_0, c_0)
        - **input** (batch, model_dim): tensor containing input features
        - **h_0** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.
        - **c_0** (batch. hidden_size): tensor containing the initial cell state
          for each element in the batch.

    Outputs: h_1, c_1
        - **h_1** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch
        - **c_1** (batch, hidden_size): tensor containing the next cell state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(4 x model_dim x hidden_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(4 x hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4 x hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4 x hidden_size)`
    """

    def __init__(self, input_size, hidden_size, bias=True, p=(0.5, 0.5)):
        super(VarLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4, input_size, hidden_size))
        self.weight_hh = Parameter(torch.Tensor(4, hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4, hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4, hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        p_in, p_hidden = p
        if p_in < 0 or p_in > 1:
            raise ValueError('input dropout probability has to be between 0 and 1, but got {}'.format(p_in))
        if p_hidden < 0 or p_hidden > 1:
            raise ValueError('hidden state dropout probability has to be between 0 and 1, but got {}'.format(p_hidden))
        self.p_in = p_in
        self.p_hidden = p_hidden
        self.noise_in = None
        self.noise_hidden = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.xavier_uniform_(self.weight_ih)
        if self.bias:
            nn.init.constant_(self.bias_hh, 0.0)
            nn.init.constant_(self.bias_ih, 0.0)

    def reset_noise(self, batch_size):
        if self.training:
            if self.p_in:
                noise = self.weight_ih.new_empty(4, batch_size, self.input_size)
                self.noise_in = noise.bernoulli_(1.0 - self.p_in) / (1.0 - self.p_in)
            else:
                self.noise_in = None
            if self.p_hidden:
                noise = self.weight_hh.new_empty(4, batch_size, self.hidden_size)
                self.noise_hidden = noise.bernoulli_(1.0 - self.p_hidden) / (1.0 - self.p_hidden)
            else:
                self.noise_hidden = None
        else:
            self.noise_in = None
            self.noise_hidden = None

    def forward(self, input, hx):
        return rnn_F.VarLSTMCell(input, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, self.noise_in, self.noise_hidden)


class VarLSTM(VarRNNBase):
    """Applies a multi-layer long short-term memory (LSTM) RNN to an input
    sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

            \\begin{array}{ll}
            i_t = \\mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\\\
            f_t = \\mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\\\
            g_t = \\tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\\\
            o_t = \\mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\\\
            c_t = f_t * c_{(t-1)} + i_t * g_t \\\\
            h_t = o_t * \\tanh(c_t)
            \\end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`x_t` is the hidden state of the previous layer at
    time `t` or :math:`input_t` for the first layer, and :math:`i_t`,
    :math:`f_t`, :math:`g_t`, :math:`o_t` are the input, forget, cell,
    and out gates, respectively.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        bias: If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: (dropout_in, dropout_hidden) tuple.
            If non-zero, introduces a dropout layer on the input and hidden of the each
            RNN layer with dropout rate dropout_in and dropout_hidden, resp.
        bidirectional: If True, becomes a bidirectional RNN. Default: False

    Inputs: input, mask, (h_0, c_0)
        - **input** (seq_len, batch, model_dim): tensor containing the features
          of the input sequence.
          **mask** (seq_len, batch): 0-1 tensor containing the mask of the input sequence.
        - **h_0** (num_layers \\* num_directions, batch, hidden_size): tensor
          containing the initial hidden state for each element in the batch.
        - **c_0** (num_layers \\* num_directions, batch, hidden_size): tensor
          containing the initial cell state for each element in the batch.

    Outputs: output, (h_n, c_n)
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for t=seq_len
        - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the cell state for t=seq_len
    """

    def __init__(self, *args, **kwargs):
        super(VarLSTM, self).__init__(VarLSTMCell, *args, **kwargs)
        self.lstm = True


class VarRNNCell(VarRNNCellBase):
    """An Elman RNN cell with tanh non-linearity and variational dropout.

    .. math::

        h' = \\tanh(w_{ih} * x + b_{ih}  +  w_{hh} * (h * \\gamma) + b_{hh})

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        nonlinearity: The non-linearity to use ['tanh'|'relu']. Default: 'tanh'
        p: (p_in, p_hidden) (tuple, optional): the drop probability for input and hidden. Default: (0.5, 0.5)

    Inputs: input, hidden
        - **input** (batch, model_dim): tensor containing input features
        - **hidden** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.

    Outputs: h'
        - **h'** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(model_dim x hidden_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(hidden_size)`

    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', p=(0.5, 0.5)):
        super(VarRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(hidden_size))
            self.bias_hh = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        p_in, p_hidden = p
        if p_in < 0 or p_in > 1:
            raise ValueError('input dropout probability has to be between 0 and 1, but got {}'.format(p_in))
        if p_hidden < 0 or p_hidden > 1:
            raise ValueError('hidden state dropout probability has to be between 0 and 1, but got {}'.format(p_hidden))
        self.p_in = p_in
        self.p_hidden = p_hidden
        self.noise_in = None
        self.noise_hidden = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.xavier_uniform_(self.weight_ih)
        if self.bias:
            nn.init.constant_(self.bias_hh, 0.0)
            nn.init.constant_(self.bias_ih, 0.0)

    def reset_noise(self, batch_size):
        if self.training:
            if self.p_in:
                noise = self.weight_ih.new_empty(batch_size, self.input_size)
                self.noise_in = noise.bernoulli_(1.0 - self.p_in) / (1.0 - self.p_in)
            else:
                self.noise_in = None
            if self.p_hidden:
                noise = self.weight_hh.new_empty(batch_size, self.hidden_size)
                self.noise_hidden = noise.bernoulli_(1.0 - self.p_hidden) / (1.0 - self.p_hidden)
            else:
                self.noise_hidden = None
        else:
            self.noise_in = None
            self.noise_hidden = None

    def forward(self, input, hx):
        if self.nonlinearity == 'tanh':
            func = rnn_F.VarRNNTanhCell
        elif self.nonlinearity == 'relu':
            func = rnn_F.VarRNNReLUCell
        else:
            raise RuntimeError('Unknown nonlinearity: {}'.format(self.nonlinearity))
        return func(input, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, self.noise_in, self.noise_hidden)


class VarRNN(VarRNNBase):
    """Applies a multi-layer Elman RNN with costomized non-linearity to an
    input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

        h_t = \\tanh(w_{ih} * x_t + b_{ih}  +  w_{hh} * h_{(t-1)} + b_{hh})

    where :math:`h_t` is the hidden state at time `t`, and :math:`x_t` is
    the hidden state of the previous layer at time `t` or :math:`input_t`
    for the first layer. If nonlinearity='relu', then `ReLU` is used instead
    of `tanh`.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        nonlinearity: The non-linearity to use ['tanh'|'relu']. Default: 'tanh'
        bias: If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: (dropout_in, dropout_hidden) tuple.
            If non-zero, introduces a dropout layer on the input and hidden of the each
            RNN layer with dropout rate dropout_in and dropout_hidden, resp.
        bidirectional: If True, becomes a bidirectional RNN. Default: False

    Inputs: input, mask, h_0
        - **input** (seq_len, batch, model_dim): tensor containing the features
          of the input sequence.
          **mask** (seq_len, batch): 0-1 tensor containing the mask of the input sequence.
        - **h_0** (num_layers * num_directions, batch, hidden_size): tensor
          containing the initial hidden state for each element in the batch.

    Outputs: output, h_n
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features (h_k) from the last layer of the RNN,
          for each k.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
          been given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for k=seq_len.
    """

    def __init__(self, *args, **kwargs):
        super(VarRNN, self).__init__(VarRNNCell, *args, **kwargs)


class DeepBiAffine(nn.Module):

    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, rnn_mode, hidden_size, num_layers, num_labels, arc_space, type_space, embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33), pos=True, activation='elu'):
        super(DeepBiAffine, self).__init__()
        self.word_embed = nn.Embedding(num_words, word_dim, _weight=embedd_word, padding_idx=1)
        self.pos_embed = nn.Embedding(num_pos, pos_dim, _weight=embedd_pos, padding_idx=1) if pos else None
        self.char_embed = nn.Embedding(num_chars, char_dim, _weight=embedd_char, padding_idx=1)
        self.char_cnn = CharCNN(2, char_dim, char_dim, hidden_channels=char_dim * 4, activation=activation)
        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)
        self.num_labels = num_labels
        if rnn_mode == 'RNN':
            RNN = VarRNN
        elif rnn_mode == 'LSTM':
            RNN = VarLSTM
        elif rnn_mode == 'FastLSTM':
            RNN = VarFastLSTM
        elif rnn_mode == 'GRU':
            RNN = VarGRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)
        dim_enc = word_dim + char_dim
        if pos:
            dim_enc += pos_dim
        self.rnn = RNN(dim_enc, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn)
        out_dim = hidden_size * 2
        self.arc_h = nn.Linear(out_dim, arc_space)
        self.arc_c = nn.Linear(out_dim, arc_space)
        self.biaffine = BiAffine(arc_space, arc_space)
        self.type_h = nn.Linear(out_dim, type_space)
        self.type_c = nn.Linear(out_dim, type_space)
        self.bilinear = BiLinear(type_space, type_space, self.num_labels)
        assert activation in ['elu', 'tanh']
        if activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.Tanh()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters(embedd_word, embedd_char, embedd_pos)

    def reset_parameters(self, embedd_word, embedd_char, embedd_pos):
        if embedd_word is None:
            nn.init.uniform_(self.word_embed.weight, -0.1, 0.1)
        if embedd_char is None:
            nn.init.uniform_(self.char_embed.weight, -0.1, 0.1)
        if embedd_pos is None and self.pos_embed is not None:
            nn.init.uniform_(self.pos_embed.weight, -0.1, 0.1)
        with torch.no_grad():
            self.word_embed.weight[self.word_embed.padding_idx].fill_(0)
            self.char_embed.weight[self.char_embed.padding_idx].fill_(0)
            if self.pos_embed is not None:
                self.pos_embed.weight[self.pos_embed.padding_idx].fill_(0)
        nn.init.xavier_uniform_(self.arc_h.weight)
        nn.init.constant_(self.arc_h.bias, 0.0)
        nn.init.xavier_uniform_(self.arc_c.weight)
        nn.init.constant_(self.arc_c.bias, 0.0)
        nn.init.xavier_uniform_(self.type_h.weight)
        nn.init.constant_(self.type_h.bias, 0.0)
        nn.init.xavier_uniform_(self.type_c.weight)
        nn.init.constant_(self.type_c.bias, 0.0)

    def _get_rnn_output(self, input_word, input_char, input_pos, mask=None):
        word = self.word_embed(input_word)
        char = self.char_cnn(self.char_embed(input_char))
        word = self.dropout_in(word)
        char = self.dropout_in(char)
        enc = torch.cat([word, char], dim=2)
        if self.pos_embed is not None:
            pos = self.pos_embed(input_pos)
            pos = self.dropout_in(pos)
            enc = torch.cat([enc, pos], dim=2)
        output, _ = self.rnn(enc, mask)
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)
        arc_h = self.activation(self.arc_h(output))
        arc_c = self.activation(self.arc_c(output))
        type_h = self.activation(self.type_h(output))
        type_c = self.activation(self.type_c(output))
        arc = torch.cat([arc_h, arc_c], dim=1)
        type = torch.cat([type_h, type_c], dim=1)
        arc = self.dropout_out(arc.transpose(1, 2)).transpose(1, 2)
        arc_h, arc_c = arc.chunk(2, 1)
        type = self.dropout_out(type.transpose(1, 2)).transpose(1, 2)
        type_h, type_c = type.chunk(2, 1)
        type_h = type_h.contiguous()
        type_c = type_c.contiguous()
        return (arc_h, arc_c), (type_h, type_c)

    def forward(self, input_word, input_char, input_pos, mask=None):
        arc, type = self._get_rnn_output(input_word, input_char, input_pos, mask=mask)
        out_arc = self.biaffine(arc[0], arc[1], mask_query=mask, mask_key=mask)
        return out_arc, type

    def loss(self, input_word, input_char, input_pos, heads, types, mask=None):
        out_arc, out_type = self(input_word, input_char, input_pos, mask=mask)
        type_h, type_c = out_type
        type_h = type_h.gather(dim=1, index=heads.unsqueeze(2).expand(type_h.size()))
        out_type = self.bilinear(type_h, type_c)
        if mask is not None:
            minus_mask = mask.eq(0).unsqueeze(2)
            out_arc = out_arc.masked_fill(minus_mask, float('-inf'))
        loss_arc = self.criterion(out_arc, heads)
        loss_type = self.criterion(out_type.transpose(1, 2), types)
        if mask is not None:
            loss_arc = loss_arc * mask
            loss_type = loss_type * mask
        return loss_arc[:, 1:].sum(dim=1), loss_type[:, 1:].sum(dim=1)

    def _decode_types(self, out_type, heads, leading_symbolic):
        type_h, type_c = out_type
        type_h = type_h.gather(dim=1, index=heads.unsqueeze(2).expand(type_h.size()))
        out_type = self.bilinear(type_h, type_c)
        out_type = out_type[:, :, leading_symbolic:]
        _, types = out_type.max(dim=2)
        return types + leading_symbolic

    def decode_local(self, input_word, input_char, input_pos, mask=None, leading_symbolic=0):
        out_arc, out_type = self(input_word, input_char, input_pos, mask=mask)
        batch, max_len, _ = out_arc.size()
        diag_mask = torch.eye(max_len, device=out_arc.device, dtype=torch.uint8).unsqueeze(0)
        out_arc.masked_fill_(diag_mask, float('-inf'))
        if mask is not None:
            minus_mask = mask.eq(0).unsqueeze(2)
            out_arc.masked_fill_(minus_mask, float('-inf'))
        _, heads = out_arc.max(dim=1)
        types = self._decode_types(out_type, heads, leading_symbolic)
        return heads.cpu().numpy(), types.cpu().numpy()

    def decode(self, input_word, input_char, input_pos, mask=None, leading_symbolic=0):
        """
        Args:
            input_word: Tensor
                the word input tensor with shape = [batch, length]
            input_char: Tensor
                the character input tensor with shape = [batch, length, char_length]
            input_pos: Tensor
                the pos input tensor with shape = [batch, length]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            length: Tensor or None
                the length tensor with shape = [batch]
            hx: Tensor or None
                the initial states of RNN
            leading_symbolic: int
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

        Returns: (Tensor, Tensor)
                predicted heads and types.

        """
        out_arc, out_type = self(input_word, input_char, input_pos, mask=mask)
        type_h, type_c = out_type
        batch, max_len, type_space = type_h.size()
        type_h = type_h.unsqueeze(2).expand(batch, max_len, max_len, type_space).contiguous()
        type_c = type_c.unsqueeze(1).expand(batch, max_len, max_len, type_space).contiguous()
        out_type = self.bilinear(type_h, type_c)
        if mask is not None:
            minus_mask = mask.eq(0).unsqueeze(2)
            out_arc.masked_fill_(minus_mask, float('-inf'))
        loss_arc = F.log_softmax(out_arc, dim=1)
        loss_type = F.log_softmax(out_type, dim=3).permute(0, 3, 1, 2)
        energy = loss_arc.unsqueeze(1) + loss_type
        length = mask.sum(dim=1).long().cpu().numpy()
        return parser.decode_MST(energy.cpu().numpy(), length, leading_symbolic=leading_symbolic, labeled=True)


class TreeCRF(nn.Module):
    """
    Tree CRF layer.
    """

    def __init__(self, model_dim):
        """

        Args:
            model_dim: int
                the dimension of the input.

        """
        super(TreeCRF, self).__init__()
        self.model_dim = model_dim
        self.energy = BiAffine(model_dim, model_dim)

    def forward(self, heads, children, mask=None):
        """

        Args:
            heads: Tensor
                the head input tensor with shape = [batch, length, model_dim]
            children: Tensor
                the child input tensor with shape = [batch, length, model_dim]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            lengths: Tensor or None
                the length tensor with shape = [batch]

        Returns: Tensor
            the energy tensor with shape = [batch, length, length]

        """
        batch, length, _ = heads.size()
        output = self.energy(heads, children, mask_query=mask, mask_key=mask)
        return output

    def loss(self, heads, children, target_heads, mask=None):
        """

        Args:
            heads: Tensor
                the head input tensor with shape = [batch, length, model_dim]
            children: Tensor
                the child input tensor with shape = [batch, length, model_dim]
            target_heads: Tensor
                the tensor of target labels with shape [batch, length]
            mask:Tensor or None
                the mask tensor with shape = [batch, length]

        Returns: Tensor
                A 1D tensor for minus log likelihood loss
        """
        batch, length, _ = heads.size()
        energy = self(heads, children, mask=mask).double()
        A = torch.exp(energy)
        if mask is not None:
            mask = mask.double()
            A = A * mask.unsqueeze(2) * mask.unsqueeze(1)
        diag_mask = 1.0 - torch.eye(length).unsqueeze(0).type_as(energy)
        A = A * diag_mask
        energy = energy * diag_mask
        D = A.sum(dim=1)
        rtol = 0.0001
        atol = 1e-06
        D += atol
        if mask is not None:
            D = D * mask
        D = torch.diag_embed(D)
        L = D - A
        if mask is not None:
            L = L + torch.diag_embed(1.0 - mask)
        L = L[:, 1:, 1:]
        z = torch.logdet(L)
        index = torch.arange(0, length).view(length, 1).expand(length, batch)
        index = index.type_as(energy).long()
        batch_index = torch.arange(0, batch).type_as(index)
        tgt_energy = energy[batch_index, target_heads.t(), index][1:]
        tgt_energy = tgt_energy.sum(dim=0)
        return (z - tgt_energy).float()


class PriorOrder(Enum):
    DEPTH = 0
    INSIDE_OUT = 1
    LEFT2RIGTH = 2


class StackPtrNet(nn.Module):

    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, rnn_mode, hidden_size, encoder_layers, decoder_layers, num_labels, arc_space, type_space, embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33), pos=True, prior_order='inside_out', grandPar=False, sibling=False, activation='elu'):
        super(StackPtrNet, self).__init__()
        self.word_embed = nn.Embedding(num_words, word_dim, _weight=embedd_word, padding_idx=1)
        self.pos_embed = nn.Embedding(num_pos, pos_dim, _weight=embedd_pos, padding_idx=1) if pos else None
        self.char_embed = nn.Embedding(num_chars, char_dim, _weight=embedd_char, padding_idx=1)
        self.char_cnn = CharCNN(2, char_dim, char_dim, hidden_channels=char_dim * 4, activation=activation)
        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)
        self.num_labels = num_labels
        if prior_order in ['deep_first', 'shallow_first']:
            self.prior_order = PriorOrder.DEPTH
        elif prior_order == 'inside_out':
            self.prior_order = PriorOrder.INSIDE_OUT
        elif prior_order == 'left2right':
            self.prior_order = PriorOrder.LEFT2RIGTH
        else:
            raise ValueError('Unknown prior order: %s' % prior_order)
        self.grandPar = grandPar
        self.sibling = sibling
        if rnn_mode == 'RNN':
            RNN_ENCODER = VarRNN
            RNN_DECODER = VarRNN
        elif rnn_mode == 'LSTM':
            RNN_ENCODER = VarLSTM
            RNN_DECODER = VarLSTM
        elif rnn_mode == 'FastLSTM':
            RNN_ENCODER = VarFastLSTM
            RNN_DECODER = VarFastLSTM
        elif rnn_mode == 'GRU':
            RNN_ENCODER = VarGRU
            RNN_DECODER = VarGRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)
        dim_enc = word_dim + char_dim
        if pos:
            dim_enc += pos_dim
        self.encoder_layers = encoder_layers
        self.encoder = RNN_ENCODER(dim_enc, hidden_size, num_layers=encoder_layers, batch_first=True, bidirectional=True, dropout=p_rnn)
        dim_dec = hidden_size // 2
        self.src_dense = nn.Linear(2 * hidden_size, dim_dec)
        self.decoder_layers = decoder_layers
        self.decoder = RNN_DECODER(dim_dec, hidden_size, num_layers=decoder_layers, batch_first=True, bidirectional=False, dropout=p_rnn)
        self.hx_dense = nn.Linear(2 * hidden_size, hidden_size)
        self.arc_h = nn.Linear(hidden_size, arc_space)
        self.arc_c = nn.Linear(hidden_size * 2, arc_space)
        self.biaffine = BiAffine(arc_space, arc_space)
        self.type_h = nn.Linear(hidden_size, type_space)
        self.type_c = nn.Linear(hidden_size * 2, type_space)
        self.bilinear = BiLinear(type_space, type_space, self.num_labels)
        assert activation in ['elu', 'tanh']
        if activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.Tanh()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters(embedd_word, embedd_char, embedd_pos)

    def reset_parameters(self, embedd_word, embedd_char, embedd_pos):
        if embedd_word is None:
            nn.init.uniform_(self.word_embed.weight, -0.1, 0.1)
        if embedd_char is None:
            nn.init.uniform_(self.char_embed.weight, -0.1, 0.1)
        if embedd_pos is None and self.pos_embed is not None:
            nn.init.uniform_(self.pos_embed.weight, -0.1, 0.1)
        with torch.no_grad():
            self.word_embed.weight[self.word_embed.padding_idx].fill_(0)
            self.char_embed.weight[self.char_embed.padding_idx].fill_(0)
            if self.pos_embed is not None:
                self.pos_embed.weight[self.pos_embed.padding_idx].fill_(0)
        nn.init.xavier_uniform_(self.arc_h.weight)
        nn.init.constant_(self.arc_h.bias, 0.0)
        nn.init.xavier_uniform_(self.arc_c.weight)
        nn.init.constant_(self.arc_c.bias, 0.0)
        nn.init.xavier_uniform_(self.type_h.weight)
        nn.init.constant_(self.type_h.bias, 0.0)
        nn.init.xavier_uniform_(self.type_c.weight)
        nn.init.constant_(self.type_c.bias, 0.0)

    def _get_encoder_output(self, input_word, input_char, input_pos, mask=None):
        word = self.word_embed(input_word)
        char = self.char_cnn(self.char_embed(input_char))
        word = self.dropout_in(word)
        char = self.dropout_in(char)
        enc = torch.cat([word, char], dim=2)
        if self.pos_embed is not None:
            pos = self.pos_embed(input_pos)
            pos = self.dropout_in(pos)
            enc = torch.cat([enc, pos], dim=2)
        output, hn = self.encoder(enc, mask)
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)
        return output, hn

    def _get_decoder_output(self, output_enc, heads, heads_stack, siblings, hx, mask=None):
        enc_dim = output_enc.size(2)
        batch, length_dec = heads_stack.size()
        src_encoding = output_enc.gather(dim=1, index=heads_stack.unsqueeze(2).expand(batch, length_dec, enc_dim))
        if self.sibling:
            mask_sib = siblings.gt(0).float().unsqueeze(2)
            output_enc_sibling = output_enc.gather(dim=1, index=siblings.unsqueeze(2).expand(batch, length_dec, enc_dim)) * mask_sib
            src_encoding = src_encoding + output_enc_sibling
        if self.grandPar:
            gpars = heads.gather(dim=1, index=heads_stack).unsqueeze(2)
            output_enc_gpar = output_enc.gather(dim=1, index=gpars.expand(batch, length_dec, enc_dim))
            src_encoding = src_encoding + output_enc_gpar
        src_encoding = self.activation(self.src_dense(src_encoding))
        output, hn = self.decoder(src_encoding, mask, hx=hx)
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)
        return output, hn

    def forward(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
        raise RuntimeError('Stack Pointer Network does not implement forward')

    def _transform_decoder_init_state(self, hn):
        if isinstance(hn, tuple):
            hn, cn = hn
            _, batch, hidden_size = cn.size()
            cn = torch.cat([cn[-2], cn[-1]], dim=1).unsqueeze(0)
            cn = self.hx_dense(cn)
            if self.decoder_layers > 1:
                cn = torch.cat([cn, cn.new_zeros(self.decoder_layers - 1, batch, hidden_size)], dim=0)
            hn = torch.tanh(cn)
            hn = hn, cn
        else:
            hn = hn[-2:]
            _, batch, hidden_size = hn.size()
            hn = hn.transpose(0, 1).contiguous()
            hn = hn.view(batch, 1, 2 * hidden_size).transpose(0, 1)
            hn = torch.tanh(self.hx_dense(hn))
            if self.decoder_layers > 1:
                hn = torch.cat([hn, hn.new_zeros(self.decoder_layers - 1, batch, hidden_size)], dim=0)
        return hn

    def loss(self, input_word, input_char, input_pos, heads, stacked_heads, children, siblings, stacked_types, mask_e=None, mask_d=None):
        output_enc, hn = self._get_encoder_output(input_word, input_char, input_pos, mask=mask_e)
        arc_c = self.activation(self.arc_c(output_enc))
        type_c = self.activation(self.type_c(output_enc))
        hn = self._transform_decoder_init_state(hn)
        output_dec, _ = self._get_decoder_output(output_enc, heads, stacked_heads, siblings, hn, mask=mask_d)
        arc_h = self.activation(self.arc_h(output_dec))
        type_h = self.activation(self.type_h(output_dec))
        batch, max_len_d, type_space = type_h.size()
        arc = self.dropout_out(torch.cat([arc_h, arc_c], dim=1).transpose(1, 2)).transpose(1, 2)
        arc_h = arc[:, :max_len_d]
        arc_c = arc[:, max_len_d:]
        type = self.dropout_out(torch.cat([type_h, type_c], dim=1).transpose(1, 2)).transpose(1, 2)
        type_h = type[:, :max_len_d].contiguous()
        type_c = type[:, max_len_d:]
        out_arc = self.biaffine(arc_h, arc_c, mask_query=mask_d, mask_key=mask_e)
        type_c = type_c.gather(dim=1, index=children.unsqueeze(2).expand(batch, max_len_d, type_space))
        out_type = self.bilinear(type_h, type_c)
        if mask_e is not None:
            minus_mask_e = mask_e.eq(0).unsqueeze(1)
            minus_mask_d = mask_d.eq(0).unsqueeze(2)
            out_arc = out_arc.masked_fill(minus_mask_d * minus_mask_e, float('-inf'))
        loss_arc = self.criterion(out_arc.transpose(1, 2), children)
        loss_type = self.criterion(out_type.transpose(1, 2), stacked_types)
        if mask_d is not None:
            loss_arc = loss_arc * mask_d
            loss_type = loss_type * mask_d
        return loss_arc.sum(dim=1), loss_type.sum(dim=1)

    def decode(self, input_word, input_char, input_pos, mask=None, beam=1, leading_symbolic=0):
        self.decoder.reset_noise(0)
        output_enc, hn = self._get_encoder_output(input_word, input_char, input_pos, mask=mask)
        enc_dim = output_enc.size(2)
        device = output_enc.device
        arc_c = self.activation(self.arc_c(output_enc))
        type_c = self.activation(self.type_c(output_enc))
        type_space = type_c.size(2)
        hn = self._transform_decoder_init_state(hn)
        batch, max_len, _ = output_enc.size()
        heads = torch.zeros(batch, 1, max_len, device=device, dtype=torch.int64)
        types = torch.zeros(batch, 1, max_len, device=device, dtype=torch.int64)
        num_steps = 2 * max_len - 1
        stacked_heads = torch.zeros(batch, 1, num_steps + 1, device=device, dtype=torch.int64)
        siblings = torch.zeros(batch, 1, num_steps + 1, device=device, dtype=torch.int64) if self.sibling else None
        hypothesis_scores = output_enc.new_zeros((batch, 1))
        children = torch.arange(max_len, device=device, dtype=torch.int64).view(1, 1, max_len).expand(batch, beam, max_len)
        constraints = torch.zeros(batch, 1, max_len, device=device, dtype=torch.bool)
        constraints[:, :, (0)] = True
        batch_index = torch.arange(batch, device=device, dtype=torch.int64).view(batch, 1)
        if mask is None:
            steps = torch.new_tensor([num_steps] * batch, dtype=torch.int64, device=device)
            mask_sent = torch.ones(batch, 1, max_len, dtype=torch.bool, device=device)
        else:
            steps = (mask.sum(dim=1) * 2 - 1).long()
            mask_sent = mask.unsqueeze(1).bool()
        num_hyp = 1
        mask_hyp = torch.ones(batch, 1, device=device)
        hx = hn
        for t in range(num_steps):
            curr_heads = stacked_heads[:, :, (t)]
            curr_gpars = heads.gather(dim=2, index=curr_heads.unsqueeze(2)).squeeze(2)
            curr_sibs = siblings[:, :, (t)] if self.sibling else None
            src_encoding = output_enc.gather(dim=1, index=curr_heads.unsqueeze(2).expand(batch, num_hyp, enc_dim))
            if self.sibling:
                mask_sib = curr_sibs.gt(0).float().unsqueeze(2)
                output_enc_sibling = output_enc.gather(dim=1, index=curr_sibs.unsqueeze(2).expand(batch, num_hyp, enc_dim)) * mask_sib
                src_encoding = src_encoding + output_enc_sibling
            if self.grandPar:
                output_enc_gpar = output_enc.gather(dim=1, index=curr_gpars.unsqueeze(2).expand(batch, num_hyp, enc_dim))
                src_encoding = src_encoding + output_enc_gpar
            src_encoding = self.activation(self.src_dense(src_encoding))
            output_dec, hx = self.decoder.step(src_encoding.view(batch * num_hyp, -1), hx=hx)
            dec_dim = output_dec.size(1)
            output_dec = output_dec.view(batch, num_hyp, dec_dim)
            arc_h = self.activation(self.arc_h(output_dec))
            type_h = self.activation(self.type_h(output_dec))
            out_arc = self.biaffine(arc_h, arc_c, mask_query=mask_hyp, mask_key=mask)
            if mask is not None:
                minus_mask_enc = mask.eq(0).unsqueeze(1)
                out_arc.masked_fill_(minus_mask_enc, float('-inf'))
            mask_last = steps.le(t + 1)
            mask_stop = steps.le(t)
            minus_mask_hyp = mask_hyp.eq(0).unsqueeze(2)
            hyp_scores = F.log_softmax(out_arc, dim=2).masked_fill_(mask_stop.view(batch, 1, 1) + minus_mask_hyp, 0)
            hypothesis_scores = hypothesis_scores.unsqueeze(2) + hyp_scores
            mask_leaf = curr_heads.unsqueeze(2).eq(children[:, :num_hyp]) * mask_sent
            mask_non_leaf = ~mask_leaf * mask_sent
            mask_leaf = mask_leaf * (mask_last.unsqueeze(1) + curr_heads.ne(0)).unsqueeze(2)
            mask_non_leaf = mask_non_leaf * ~constraints
            hypothesis_scores.masked_fill_(~(mask_non_leaf + mask_leaf), float('-inf'))
            hypothesis_scores, hyp_index = torch.sort(hypothesis_scores.view(batch, -1), dim=1, descending=True)
            prev_num_hyp = num_hyp
            num_hyps = (mask_leaf + mask_non_leaf).long().view(batch, -1).sum(dim=1)
            num_hyp = num_hyps.max().clamp(max=beam).item()
            hyps = torch.arange(num_hyp, device=device, dtype=torch.int64).view(1, num_hyp)
            mask_hyp = hyps.lt(num_hyps.unsqueeze(1)).float()
            hypothesis_scores = hypothesis_scores[:, :num_hyp]
            hyp_index = hyp_index[:, :num_hyp]
            base_index = hyp_index / max_len
            child_index = hyp_index % max_len
            hyp_heads = curr_heads.gather(dim=1, index=base_index)
            hyp_gpars = curr_gpars.gather(dim=1, index=base_index)
            base_index_expand = base_index.unsqueeze(2).expand(batch, num_hyp, max_len)
            constraints = constraints.gather(dim=1, index=base_index_expand)
            constraints.scatter_(2, child_index.unsqueeze(2), True)
            mask_leaf = hyp_heads.eq(child_index)
            heads = heads.gather(dim=1, index=base_index_expand)
            heads.scatter_(2, child_index.unsqueeze(2), torch.where(mask_leaf, hyp_gpars, hyp_heads).unsqueeze(2))
            types = types.gather(dim=1, index=base_index_expand)
            org_types = types.gather(dim=2, index=child_index.unsqueeze(2)).squeeze(2)
            base_index_expand = base_index.unsqueeze(2).expand(batch, num_hyp, num_steps + 1)
            stacked_heads = stacked_heads.gather(dim=1, index=base_index_expand)
            stacked_heads[:, :, (t + 1)] = torch.where(mask_leaf, hyp_gpars, child_index)
            if self.sibling:
                siblings = siblings.gather(dim=1, index=base_index_expand)
                siblings[:, :, (t + 1)] = torch.where(mask_leaf, child_index, torch.zeros_like(child_index))
            base_index_expand = base_index.unsqueeze(2).expand(batch, num_hyp, type_space)
            child_index_expand = child_index.unsqueeze(2).expand(batch, num_hyp, type_space)
            out_type = self.bilinear(type_h.gather(dim=1, index=base_index_expand), type_c.gather(dim=1, index=child_index_expand))
            hyp_type_scores = F.log_softmax(out_type, dim=2)
            hyp_type_scores, hyp_types = hyp_type_scores.max(dim=2)
            hypothesis_scores = hypothesis_scores + hyp_type_scores.masked_fill_(mask_stop.view(batch, 1), 0)
            types.scatter_(2, child_index.unsqueeze(2), torch.where(mask_leaf, org_types, hyp_types).unsqueeze(2))
            hx_index = (base_index + batch_index * prev_num_hyp).view(batch * num_hyp)
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx[:, (hx_index)]
                cx = cx[:, (hx_index)]
                hx = hx, cx
            else:
                hx = hx[:, (hx_index)]
        heads = heads[:, (0)].cpu().numpy()
        types = types[:, (0)].cpu().numpy()
        return heads, types


class BiRecurrentConv(nn.Module):

    def __init__(self, word_dim, num_words, char_dim, num_chars, rnn_mode, hidden_size, out_features, num_layers, num_labels, embedd_word=None, embedd_char=None, p_in=0.33, p_out=0.5, p_rnn=(0.5, 0.5), activation='elu'):
        super(BiRecurrentConv, self).__init__()
        self.word_embed = nn.Embedding(num_words, word_dim, _weight=embedd_word, padding_idx=1)
        self.char_embed = nn.Embedding(num_chars, char_dim, _weight=embedd_char, padding_idx=1)
        self.char_cnn = CharCNN(2, char_dim, char_dim, hidden_channels=4 * char_dim, activation=activation)
        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_rnn_in = nn.Dropout(p=p_rnn[0])
        self.dropout_out = nn.Dropout(p_out)
        if rnn_mode == 'RNN':
            RNN = nn.RNN
        elif rnn_mode == 'LSTM' or rnn_mode == 'FastLSTM':
            RNN = nn.LSTM
        elif rnn_mode == 'GRU':
            RNN = nn.GRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)
        self.rnn = RNN(word_dim + char_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn[1])
        self.fc = nn.Linear(hidden_size * 2, out_features)
        assert activation in ['elu', 'tanh']
        if activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.Tanh()
        self.readout = nn.Linear(out_features, num_labels)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters(embedd_word, embedd_char)

    def reset_parameters(self, embedd_word, embedd_char):
        if embedd_word is None:
            nn.init.uniform_(self.word_embed.weight, -0.1, 0.1)
        if embedd_char is None:
            nn.init.uniform_(self.char_embed.weight, -0.1, 0.1)
        with torch.no_grad():
            self.word_embed.weight[self.word_embed.padding_idx].fill_(0)
            self.char_embed.weight[self.char_embed.padding_idx].fill_(0)
        for param in self.rnn.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        nn.init.uniform_(self.readout.weight, -0.1, 0.1)
        nn.init.constant_(self.readout.bias, 0.0)

    def _get_rnn_output(self, input_word, input_char, mask=None):
        word = self.word_embed(input_word)
        char = self.char_cnn(self.char_embed(input_char))
        word = self.dropout_in(word)
        char = self.dropout_in(char)
        enc = torch.cat([word, char], dim=2)
        enc = self.dropout_rnn_in(enc)
        if mask is not None:
            length = mask.sum(dim=1).long()
            packed_enc = pack_padded_sequence(enc, length, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.rnn(packed_enc)
            output, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            output, _ = self.rnn(enc)
        output = self.dropout_out(output)
        output = self.dropout_out(self.activation(self.fc(output)))
        return output

    def forward(self, input_word, input_char, mask=None):
        output = self._get_rnn_output(input_word, input_char, mask=mask)
        return output

    def loss(self, input_word, input_char, target, mask=None):
        output = self(input_word, input_char, mask=mask)
        logits = self.readout(output).transpose(1, 2)
        loss = self.criterion(logits, target)
        if mask is not None:
            loss = loss * mask
        loss = loss.sum(dim=1)
        return loss

    def decode(self, input_word, input_char, mask=None, leading_symbolic=0):
        output = self(input_word, input_char, mask=mask)
        logits = self.readout(output).transpose(1, 2)
        _, preds = torch.max(logits[:, leading_symbolic:], dim=1)
        preds += leading_symbolic
        if mask is not None:
            preds = preds * mask.long()
        return preds


class ChainCRF(nn.Module):

    def __init__(self, input_size, num_labels, bigram=True):
        """

        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            bigram: bool
                if apply bi-gram parameter.
        """
        super(ChainCRF, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels + 1
        self.pad_label_id = num_labels
        self.bigram = bigram
        self.state_net = nn.Linear(input_size, self.num_labels)
        if bigram:
            self.transition_net = nn.Linear(input_size, self.num_labels * self.num_labels)
            self.register_parameter('transition_matrix', None)
        else:
            self.transition_net = None
            self.transition_matrix = Parameter(torch.Tensor(self.num_labels, self.num_labels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.state_net.bias, 0.0)
        if self.bigram:
            nn.init.xavier_uniform_(self.transition_net.weight)
            nn.init.constant_(self.transition_net.bias, 0.0)
        else:
            nn.init.normal_(self.transition_matrix)

    def forward(self, input, mask=None):
        """

        Args:
            input: Tensor
                the input tensor with shape = [batch, length, model_dim]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]

        Returns: Tensor
            the energy tensor with shape = [batch, length, num_label, num_label]

        """
        batch, length, _ = input.size()
        out_s = self.state_net(input).unsqueeze(2)
        if self.bigram:
            out_t = self.transition_net(input).view(batch, length, self.num_labels, self.num_labels)
            output = out_t + out_s
        else:
            output = self.transition_matrix + out_s
        if mask is not None:
            output = output * mask.unsqueeze(2).unsqueeze(3)
        return output

    def loss(self, input, target, mask=None):
        """

        Args:
            input: Tensor
                the input tensor with shape = [batch, length, model_dim]
            target: Tensor
                the tensor of target labels with shape [batch, length]
            mask:Tensor or None
                the mask tensor with shape = [batch, length]

        Returns: Tensor
                A 1D tensor for minus log likelihood loss [batch]
        """
        batch, length, _ = input.size()
        energy = self(input, mask=mask)
        energy_transpose = energy.transpose(0, 1)
        target_transpose = target.transpose(0, 1)
        mask_transpose = None
        if mask is not None:
            mask_transpose = mask.unsqueeze(2).transpose(0, 1)
        partition = None
        batch_index = torch.arange(0, batch).type_as(input).long()
        prev_label = input.new_full((batch,), self.num_labels - 1).long()
        tgt_energy = input.new_zeros(batch)
        for t in range(length):
            curr_energy = energy_transpose[t]
            if t == 0:
                partition = curr_energy[:, (-1), :]
            else:
                partition_new = torch.logsumexp(curr_energy + partition.unsqueeze(2), dim=1)
                if mask_transpose is None:
                    partition = partition_new
                else:
                    mask_t = mask_transpose[t]
                    partition = partition + (partition_new - partition) * mask_t
            tgt_energy += curr_energy[batch_index, prev_label, target_transpose[t]]
            prev_label = target_transpose[t]
        return torch.logsumexp(partition, dim=1) - tgt_energy

    def decode(self, input, mask=None, leading_symbolic=0):
        """

        Args:
            input: Tensor
                the input tensor with shape = [batch, length, model_dim]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            leading_symbolic: nt
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

        Returns: Tensor
            decoding results in shape [batch, length]

        """
        energy = self(input, mask=mask)
        energy_transpose = energy.transpose(0, 1)
        energy_transpose = energy_transpose[:, :, leading_symbolic:-1, leading_symbolic:-1]
        length, batch_size, num_label, _ = energy_transpose.size()
        batch_index = torch.arange(0, batch_size).type_as(input).long()
        pi = input.new_zeros([length, batch_size, num_label])
        pointer = batch_index.new_zeros(length, batch_size, num_label)
        back_pointer = batch_index.new_zeros(length, batch_size)
        pi[0] = energy[:, (0), (-1), leading_symbolic:-1]
        pointer[0] = -1
        for t in range(1, length):
            pi_prev = pi[t - 1]
            pi[t], pointer[t] = torch.max(energy_transpose[t] + pi_prev.unsqueeze(2), dim=1)
        _, back_pointer[-1] = torch.max(pi[-1], dim=1)
        for t in reversed(range(length - 1)):
            pointer_last = pointer[t + 1]
            back_pointer[t] = pointer_last[batch_index, back_pointer[t + 1]]
        return back_pointer.transpose(0, 1) + leading_symbolic


class VarSkipRNNBase(nn.Module):

    def __init__(self, Cell, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=(0, 0), bidirectional=False, **kwargs):
        super(VarSkipRNNBase, self).__init__()
        self.Cell = Cell
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.lstm = False
        num_directions = 2 if bidirectional else 1
        self.all_cells = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                cell = self.Cell(layer_input_size, hidden_size, self.bias, p=dropout, **kwargs)
                self.all_cells.append(cell)
                self.add_module('cell%d' % (layer * num_directions + direction), cell)

    def reset_parameters(self):
        for cell in self.all_cells:
            cell.reset_parameters()

    def reset_noise(self, batch_size):
        for cell in self.all_cells:
            cell.reset_noise(batch_size)

    def forward(self, input, skip_connect, mask=None, hx=None):
        batch_size = input.size(0) if self.batch_first else input.size(1)
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size)
            if self.lstm:
                hx = hx, hx
        func = rnn_F.AutogradSkipConnectRNN(num_layers=self.num_layers, batch_first=self.batch_first, bidirectional=self.bidirectional, lstm=self.lstm)
        self.reset_noise(batch_size)
        output, hidden = func(input, skip_connect, self.all_cells, hx, None if mask is None else mask.view(mask.size() + (1,)))
        return output, hidden

    def step(self, input, hx=None, hs=None, mask=None):
        """
        execute one step forward (only for one-directional RNN).
        Args:
            input (batch, model_dim): input tensor of this step.
            hx (num_layers, batch, hidden_size): the hidden state of last step.
            hs (batch. hidden_size): tensor containing the skip connection state for each element in the batch.
            mask (batch): the mask tensor of this step.

        Returns:
            output (batch, hidden_size): tensor containing the output of this step from the last layer of RNN.
            hn (num_layers, batch, hidden_size): tensor containing the hidden state of this step
        """
        assert not self.bidirectional, 'step only cannot be applied to bidirectional RNN.'
        batch_size = input.size(0)
        if hx is None:
            hx = input.new_zeros(self.num_layers, batch_size, self.hidden_size)
            if self.lstm:
                hx = hx, hx
        if hs is None:
            hs = input.new_zeros(self.num_layers, batch_size, self.hidden_size)
        func = rnn_F.AutogradSkipConnectStep(num_layers=self.num_layers, lstm=self.lstm)
        output, hidden = func(input, self.all_cells, hx, hs, mask)
        return output, hidden


class SkipConnectRNNCell(VarRNNCellBase):
    """An Elman RNN cell with tanh non-linearity and variational dropout.

    .. math::

        h' = \\tanh(w_{ih} * x + b_{ih}  +  w_{hh} * (h * \\gamma) + b_{hh})

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        nonlinearity: The non-linearity to use ['tanh'|'relu']. Default: 'tanh'
        p: (p_in, p_hidden) (tuple, optional): the drop probability for input and hidden. Default: (0.5, 0.5)

    Inputs: input, hidden, h_s
        - **input** (batch, model_dim): tensor containing input features
        - **hidden** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.
        - **h_s** (batch. hidden_size): tensor containing the skip connection state
          for each element in the batch.

    Outputs: h'
        - **h'** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(model_dim x hidden_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(hidden_size x 2*hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(hidden_size)`

    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', p=(0.5, 0.5)):
        super(SkipConnectRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.weight_ih = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(hidden_size, hidden_size * 2))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(hidden_size))
            self.bias_hh = Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        p_in, p_hidden = p
        if p_in < 0 or p_in > 1:
            raise ValueError('input dropout probability has to be between 0 and 1, but got {}'.format(p_in))
        if p_hidden < 0 or p_hidden > 1:
            raise ValueError('hidden state dropout probability has to be between 0 and 1, but got {}'.format(p_hidden))
        self.p_in = p_in
        self.p_hidden = p_hidden
        self.noise_in = None
        self.noise_hidden = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.xavier_uniform_(self.weight_ih)
        if self.bias:
            nn.init.constant_(self.bias_hh, 0.0)
            nn.init.constant_(self.bias_ih, 0.0)

    def reset_noise(self, batch_size):
        if self.training:
            if self.p_in:
                noise = self.weight_ih.new_empty(batch_size, self.input_size)
                self.noise_in = noise.bernoulli_(1.0 - self.p_in) / (1.0 - self.p_in)
            else:
                self.noise_in = None
            if self.p_hidden:
                noise = self.weight_hh.new_empty(batch_size, self.hidden_size * 2)
                self.noise_hidden = noise.bernoulli_(1.0 - self.p_hidden) / (1.0 - self.p_hidden)
            else:
                self.noise_hidden = None
        else:
            self.noise_in = None
            self.noise_hidden = None

    def forward(self, input, hx, hs):
        if self.nonlinearity == 'tanh':
            func = rnn_F.SkipConnectRNNTanhCell
        elif self.nonlinearity == 'relu':
            func = rnn_F.SkipConnectRNNReLUCell
        else:
            raise RuntimeError('Unknown nonlinearity: {}'.format(self.nonlinearity))
        return func(input, hx, hs, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, self.noise_in, self.noise_hidden)


class VarSkipRNN(VarSkipRNNBase):
    """Applies a multi-layer Elman RNN with costomized non-linearity to an
    input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

        h_t = \\tanh(w_{ih} * x_t + b_{ih}  +  w_{hh} * h_{(t-1)} + b_{hh})

    where :math:`h_t` is the hidden state at time `t`, and :math:`x_t` is
    the hidden state of the previous layer at time `t` or :math:`input_t`
    for the first layer. If nonlinearity='relu', then `ReLU` is used instead
    of `tanh`.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        nonlinearity: The non-linearity to use ['tanh'|'relu']. Default: 'tanh'
        bias: If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: (dropout_in, dropout_hidden) tuple.
            If non-zero, introduces a dropout layer on the input and hidden of the each
            RNN layer with dropout rate dropout_in and dropout_hidden, resp.
        bidirectional: If True, becomes a bidirectional RNN. Default: False

    Inputs: input, skip_connect, mask, h_0
        - **input** (seq_len, batch, model_dim): tensor containing the features
          of the input sequence.
        - **skip_connect** (seq_len, batch): long tensor containing the index of skip connections for each step.
          **mask** (seq_len, batch): 0-1 tensor containing the mask of the input sequence.
        - **h_0** (num_layers * num_directions, batch, hidden_size): tensor
          containing the initial hidden state for each element in the batch.

    Outputs: output, h_n
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features (h_k) from the last layer of the RNN,
          for each k.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
          been given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for k=seq_len.
    """

    def __init__(self, *args, **kwargs):
        super(VarSkipRNN, self).__init__(SkipConnectRNNCell, *args, **kwargs)


class SkipConnectFastLSTMCell(VarRNNCellBase):
    """
    A long short-term memory (LSTM) cell with skip connections and variational dropout.

    .. math::

        egin{array}{ll}
        i = \\mathrm{sigmoid}(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \\mathrm{sigmoid}(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = 	anh(W_{ig} x + b_{ig} + W_{hc} h + b_{hg}) \\
        o = \\mathrm{sigmoid}(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * 	anh(c') \\
        \\end{array}

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: True
        p: (p_in, p_hidden) (tuple, optional): the drop probability for input and hidden. Default: (0.5, 0.5)

    Inputs: input, (h_0, c_0), h_s
        - **input** (batch, model_dim): tensor containing input features
        - **h_0** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.
        - **c_0** (batch. hidden_size): tensor containing the initial cell state
          for each element in the batch.
        - **h_s** (batch. hidden_size): tensor containing the skip connection state
          for each element in the batch.

    Outputs: h_1, c_1
        - **h_1** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch
        - **c_1** (batch, hidden_size): tensor containing the next cell state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(4*hidden_size x model_dim)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(4*hidden_size x 2*hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`
    """

    def __init__(self, input_size, hidden_size, bias=True, p=(0.5, 0.5)):
        super(SkipConnectFastLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, 2 * hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        p_in, p_hidden = p
        if p_in < 0 or p_in > 1:
            raise ValueError('input dropout probability has to be between 0 and 1, but got {}'.format(p_in))
        if p_hidden < 0 or p_hidden > 1:
            raise ValueError('hidden state dropout probability has to be between 0 and 1, but got {}'.format(p_hidden))
        self.p_in = p_in
        self.p_hidden = p_hidden
        self.noise_in = None
        self.noise_hidden = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.xavier_uniform_(self.weight_ih)
        if self.bias:
            nn.init.constant_(self.bias_hh, 0.0)
            nn.init.constant_(self.bias_ih, 0.0)

    def reset_noise(self, batch_size):
        if self.training:
            if self.p_in:
                noise = self.weight_ih.new_empty(batch_size, self.input_size)
                self.noise_in = noise.bernoulli_(1.0 - self.p_in) / (1.0 - self.p_in)
            else:
                self.noise_in = None
            if self.p_hidden:
                noise = self.weight_hh.new_empty(batch_size, self.hidden_size * 2)
                self.noise_hidden = noise.bernoulli_(1.0 - self.p_hidden) / (1.0 - self.p_hidden)
            else:
                self.noise_hidden = None
        else:
            self.noise_in = None
            self.noise_hidden = None

    def forward(self, input, hx, hs):
        return rnn_F.SkipConnectFastLSTMCell(input, hx, hs, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, self.noise_in, self.noise_hidden)


class VarSkipFastLSTM(VarSkipRNNBase):
    """Applies a multi-layer long short-term memory (LSTM) RNN to an input
    sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

            \\begin{array}{ll}
            i_t = \\mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\\\
            f_t = \\mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\\\
            g_t = \\tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\\\
            o_t = \\mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\\\
            c_t = f_t * c_{(t-1)} + i_t * g_t \\\\
            h_t = o_t * \\tanh(c_t)
            \\end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`x_t` is the hidden state of the previous layer at
    time `t` or :math:`input_t` for the first layer, and :math:`i_t`,
    :math:`f_t`, :math:`g_t`, :math:`o_t` are the input, forget, cell,
    and out gates, respectively.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        bias: If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: (dropout_in, dropout_hidden) tuple.
            If non-zero, introduces a dropout layer on the input and hidden of the each
            RNN layer with dropout rate dropout_in and dropout_hidden, resp.
        bidirectional: If True, becomes a bidirectional RNN. Default: False

    Inputs: input, skip_connect, mask, (h_0, c_0)
        - **input** (seq_len, batch, model_dim): tensor containing the features
          of the input sequence.
        - **skip_connect** (seq_len, batch): long tensor containing the index of skip connections for each step.
        - **mask** (seq_len, batch): 0-1 tensor containing the mask of the input sequence.
        - **h_0** (num_layers \\* num_directions, batch, hidden_size): tensor
          containing the initial hidden state for each element in the batch.
        - **c_0** (num_layers \\* num_directions, batch, hidden_size): tensor
          containing the initial cell state for each element in the batch.

    Outputs: output, (h_n, c_n)
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for t=seq_len
        - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the cell state for t=seq_len
    """

    def __init__(self, *args, **kwargs):
        super(VarSkipFastLSTM, self).__init__(SkipConnectFastLSTMCell, *args, **kwargs)
        self.lstm = True


class SkipConnectLSTMCell(VarRNNCellBase):
    """
    A long short-term memory (LSTM) cell with skip connections and variational dropout.

    .. math::

        egin{array}{ll}
        i = \\mathrm{sigmoid}(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \\mathrm{sigmoid}(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = 	anh(W_{ig} x + b_{ig} + W_{hc} h + b_{hg}) \\
        o = \\mathrm{sigmoid}(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * 	anh(c') \\
        \\end{array}

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: True
        p: (p_in, p_hidden) (tuple, optional): the drop probability for input and hidden. Default: (0.5, 0.5)

    Inputs: input, (h_0, c_0), h_s
        - **input** (batch, model_dim): tensor containing input features
        - **h_0** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.
        - **c_0** (batch. hidden_size): tensor containing the initial cell state
          for each element in the batch.
           **h_s** (batch. hidden_size): tensor containing the skip connection state
          for each element in the batch.

    Outputs: h_1, c_1
        - **h_1** (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch
        - **c_1** (batch, hidden_size): tensor containing the next cell state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(4 x model_dim x hidden_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(4 x 2*hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4 x hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4 x hidden_size)`
    """

    def __init__(self, input_size, hidden_size, bias=True, p=(0.5, 0.5)):
        super(SkipConnectLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4, input_size, hidden_size))
        self.weight_hh = Parameter(torch.Tensor(4, 2 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4, hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4, hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        p_in, p_hidden = p
        if p_in < 0 or p_in > 1:
            raise ValueError('input dropout probability has to be between 0 and 1, but got {}'.format(p_in))
        if p_hidden < 0 or p_hidden > 1:
            raise ValueError('hidden state dropout probability has to be between 0 and 1, but got {}'.format(p_hidden))
        self.p_in = p_in
        self.p_hidden = p_hidden
        self.noise_in = None
        self.noise_hidden = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.xavier_uniform_(self.weight_ih)
        if self.bias:
            nn.init.constant_(self.bias_hh, 0.0)
            nn.init.constant_(self.bias_ih, 0.0)

    def reset_noise(self, batch_size):
        if self.training:
            if self.p_in:
                noise = self.weight_ih.new_empty(4, batch_size, self.input_size)
                self.noise_in = noise.bernoulli_(1.0 - self.p_in) / (1.0 - self.p_in)
            else:
                self.noise_in = None
            if self.p_hidden:
                noise = self.weight_hh.new_empty(4, batch_size, self.hidden_size * 2)
                self.noise_hidden = noise.bernoulli_(1.0 - self.p_hidden) / (1.0 - self.p_hidden)
            else:
                self.noise_hidden = None
        else:
            self.noise_in = None
            self.noise_hidden = None

    def forward(self, input, hx, hs):
        return rnn_F.SkipConnectLSTMCell(input, hx, hs, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, self.noise_in, self.noise_hidden)


class VarSkipLSTM(VarSkipRNNBase):
    """Applies a multi-layer long short-term memory (LSTM) RNN to an input
    sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

            \\begin{array}{ll}
            i_t = \\mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\\\
            f_t = \\mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\\\
            g_t = \\tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\\\
            o_t = \\mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\\\
            c_t = f_t * c_{(t-1)} + i_t * g_t \\\\
            h_t = o_t * \\tanh(c_t)
            \\end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`x_t` is the hidden state of the previous layer at
    time `t` or :math:`input_t` for the first layer, and :math:`i_t`,
    :math:`f_t`, :math:`g_t`, :math:`o_t` are the input, forget, cell,
    and out gates, respectively.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        bias: If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: (dropout_in, dropout_hidden) tuple.
            If non-zero, introduces a dropout layer on the input and hidden of the each
            RNN layer with dropout rate dropout_in and dropout_hidden, resp.
        bidirectional: If True, becomes a bidirectional RNN. Default: False

    Inputs: input, skip_connect, mask, (h_0, c_0)
        - **input** (seq_len, batch, model_dim): tensor containing the features
          of the input sequence.
        - **skip_connect** (seq_len, batch): long tensor containing the index of skip connections for each step.
        - **mask** (seq_len, batch): 0-1 tensor containing the mask of the input sequence.
        - **h_0** (num_layers \\* num_directions, batch, hidden_size): tensor
          containing the initial hidden state for each element in the batch.
        - **c_0** (num_layers \\* num_directions, batch, hidden_size): tensor
          containing the initial cell state for each element in the batch.

    Outputs: output, (h_n, c_n)
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for t=seq_len
        - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the cell state for t=seq_len
    """

    def __init__(self, *args, **kwargs):
        super(VarSkipLSTM, self).__init__(SkipConnectLSTMCell, *args, **kwargs)
        self.lstm = True


class SkipConnectFastGRUCell(VarRNNCellBase):
    """A gated recurrent unit (GRU) cell with skip connections and variational dropout.

    .. math::

        egin{array}{ll}
        r = \\mathrm{sigmoid}(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \\mathrm{sigmoid}(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = 	anh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \\end{array}

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: True
        p: (p_in, p_hidden) (tuple, optional): the drop probability for input and hidden. Default: (0.5, 0.5)

    Inputs: input, hidden, h_s
        - **input** (batch, model_dim): tensor containing input features
        - **hidden** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.
        - **h_s** (batch. hidden_size): tensor containing the skip connection state
          for each element in the batch.

    Outputs: h'
        - **h'**: (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(3*hidden_size x model_dim)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(3*hidden_size x 2*hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(3*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(3*hidden_size)`
    """

    def __init__(self, input_size, hidden_size, bias=True, p=(0.5, 0.5)):
        super(SkipConnectFastGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size * 2))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        p_in, p_hidden = p
        if p_in < 0 or p_in > 1:
            raise ValueError('input dropout probability has to be between 0 and 1, but got {}'.format(p_in))
        if p_hidden < 0 or p_hidden > 1:
            raise ValueError('hidden state dropout probability has to be between 0 and 1, but got {}'.format(p_hidden))
        self.p_in = p_in
        self.p_hidden = p_hidden
        self.noise_in = None
        self.noise_hidden = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.xavier_uniform_(self.weight_ih)
        if self.bias:
            nn.init.constant_(self.bias_hh, 0.0)
            nn.init.constant_(self.bias_ih, 0.0)

    def reset_noise(self, batch_size):
        if self.training:
            if self.p_in:
                noise = self.weight_ih.new_empty(batch_size, self.input_size)
                self.noise_in = noise.bernoulli_(1.0 - self.p_in) / (1.0 - self.p_in)
            else:
                self.noise_in = None
            if self.p_hidden:
                noise = self.weight_hh.new_empty(batch_size, self.hidden_size * 2)
                self.noise_hidden = noise.bernoulli_(1.0 - self.p_hidden) / (1.0 - self.p_hidden)
            else:
                self.noise_hidden = None
        else:
            self.noise_in = None
            self.noise_hidden = None

    def forward(self, input, hx, hs):
        return rnn_F.SkipConnectFastGRUCell(input, hx, hs, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, self.noise_in, self.noise_hidden)


class VarSkipFastGRU(VarSkipRNNBase):
    """Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

            \\begin{array}{ll}
            r_t = \\mathrm{sigmoid}(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\\\
            z_t = \\mathrm{sigmoid}(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\\\
            n_t = \\tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\\\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)} \\\\
            \\end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the hidden
    state of the previous layer at time `t` or :math:`input_t` for the first
    layer, and :math:`r_t`, :math:`z_t`, :math:`n_t` are the reset, input,
    and new gates, respectively.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        nonlinearity: The non-linearity to use ['tanh'|'relu']. Default: 'tanh'
        bias: If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: (dropout_in, dropout_hidden) tuple.
            If non-zero, introduces a dropout layer on the input and hidden of the each
            RNN layer with dropout rate dropout_in and dropout_hidden, resp.
        bidirectional: If True, becomes a bidirectional RNN. Default: False

    Inputs: input, skip_connect, mask, h_0
        - **input** (seq_len, batch, model_dim): tensor containing the features
          of the input sequence.
        - **skip_connect** (seq_len, batch): long tensor containing the index of skip connections for each step.
          **mask** (seq_len, batch): 0-1 tensor containing the mask of the input sequence.
        - **h_0** (num_layers * num_directions, batch, hidden_size): tensor
          containing the initial hidden state for each element in the batch.

    Outputs: output, h_n
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features (h_k) from the last layer of the RNN,
          for each k.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
          been given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for k=seq_len.
    """

    def __init__(self, *args, **kwargs):
        super(VarSkipFastGRU, self).__init__(SkipConnectFastGRUCell, *args, **kwargs)


class SkipConnectGRUCell(VarRNNCellBase):
    """A gated recurrent unit (GRU) cell with skip connections and variational dropout.

    .. math::

        egin{array}{ll}
        r = \\mathrm{sigmoid}(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \\mathrm{sigmoid}(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = 	anh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \\end{array}

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: `True`
        p: (p_in, p_hidden) (tuple, optional): the drop probability for input and hidden. Default: (0.5, 0.5)

    Inputs: input, hidden, h_s
        - **input** (batch, model_dim): tensor containing input features
        - **hidden** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.
        - **h_s** (batch. hidden_size): tensor containing the skip connection state
          for each element in the batch.

    Outputs: h'
        - **h'**: (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(3 x model_dim x hidden_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(3x 2*hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(3 x hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(3 x hidden_size)`
    """

    def __init__(self, input_size, hidden_size, bias=True, p=(0.5, 0.5)):
        super(SkipConnectGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3, input_size, hidden_size))
        self.weight_hh = Parameter(torch.Tensor(3, hidden_size * 2, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3, hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3, hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        p_in, p_hidden = p
        if p_in < 0 or p_in > 1:
            raise ValueError('input dropout probability has to be between 0 and 1, but got {}'.format(p_in))
        if p_hidden < 0 or p_hidden > 1:
            raise ValueError('hidden state dropout probability has to be between 0 and 1, but got {}'.format(p_hidden))
        self.p_in = p_in
        self.p_hidden = p_hidden
        self.noise_in = None
        self.noise_hidden = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.xavier_uniform_(self.weight_ih)
        if self.bias:
            nn.init.constant_(self.bias_hh, 0.0)
            nn.init.constant_(self.bias_ih, 0.0)

    def reset_noise(self, batch_size):
        if self.training:
            if self.p_in:
                noise = self.weight_ih.new_empty(3, batch_size, self.input_size)
                self.noise_in = noise.bernoulli_(1.0 - self.p_in) / (1.0 - self.p_in)
            else:
                self.noise_in = None
            if self.p_hidden:
                noise = self.weight_hh.new_empty(3, batch_size, self.hidden_size * 2)
                self.noise_hidden = noise.bernoulli_(1.0 - self.p_hidden) / (1.0 - self.p_hidden)
            else:
                self.noise_hidden = None
        else:
            self.noise_in = None
            self.noise_hidden = None

    def forward(self, input, hx, hs):
        return rnn_F.SkipConnectGRUCell(input, hx, hs, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, self.noise_in, self.noise_hidden)


class VarSkipGRU(VarSkipRNNBase):
    """Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

            \\begin{array}{ll}
            r_t = \\mathrm{sigmoid}(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\\\
            z_t = \\mathrm{sigmoid}(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\\\
            n_t = \\tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\\\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)} \\\\
            \\end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the hidden
    state of the previous layer at time `t` or :math:`input_t` for the first
    layer, and :math:`r_t`, :math:`z_t`, :math:`n_t` are the reset, input,
    and new gates, respectively.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        nonlinearity: The non-linearity to use ['tanh'|'relu']. Default: 'tanh'
        bias: If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: (dropout_in, dropout_hidden) tuple.
            If non-zero, introduces a dropout layer on the input and hidden of the each
            RNN layer with dropout rate dropout_in and dropout_hidden, resp.
        bidirectional: If True, becomes a bidirectional RNN. Default: False

    Inputs: input, skip_connect, mask, h_0
        - **input** (seq_len, batch, model_dim): tensor containing the features
          of the input sequence.
        - **skip_connect** (seq_len, batch): long tensor containing the index of skip connections for each step.
          **mask** (seq_len, batch): 0-1 tensor containing the mask of the input sequence.
        - **h_0** (num_layers * num_directions, batch, hidden_size): tensor
          containing the initial hidden state for each element in the batch.

    Outputs: output, h_n
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features (h_k) from the last layer of the RNN,
          for each k.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
          been given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for k=seq_len.
    """

    def __init__(self, *args, **kwargs):
        super(VarSkipGRU, self).__init__(SkipConnectGRUCell, *args, **kwargs)


class VarFastGRUCell(VarRNNCellBase):
    """A gated recurrent unit (GRU) cell with variational dropout.

    .. math::

        egin{array}{ll}
        r = \\mathrm{sigmoid}(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \\mathrm{sigmoid}(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = 	anh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \\end{array}

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: True
        p: (p_in, p_hidden) (tuple, optional): the drop probability for input and hidden. Default: (0.5, 0.5)

    Inputs: input, hidden
        - **input** (batch, model_dim): tensor containing input features
        - **hidden** (batch, hidden_size): tensor containing the initial hidden
          state for each element in the batch.

    Outputs: h'
        - **h'**: (batch, hidden_size): tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(3*hidden_size x model_dim)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(3*hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(3*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(3*hidden_size)`
    """

    def __init__(self, input_size, hidden_size, bias=True, p=(0.5, 0.5)):
        super(VarFastGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        p_in, p_hidden = p
        if p_in < 0 or p_in > 1:
            raise ValueError('input dropout probability has to be between 0 and 1, but got {}'.format(p_in))
        if p_hidden < 0 or p_hidden > 1:
            raise ValueError('hidden state dropout probability has to be between 0 and 1, but got {}'.format(p_hidden))
        self.p_in = p_in
        self.p_hidden = p_hidden
        self.noise_in = None
        self.noise_hidden = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.xavier_uniform_(self.weight_ih)
        if self.bias:
            nn.init.constant_(self.bias_hh, 0.0)
            nn.init.constant_(self.bias_ih, 0.0)

    def reset_noise(self, batch_size):
        if self.training:
            if self.p_in:
                noise = self.weight_ih.new_empty(batch_size, self.input_size)
                self.noise_in = noise.bernoulli_(1.0 - self.p_in) / (1.0 - self.p_in)
            else:
                self.noise_in = None
            if self.p_hidden:
                noise = self.weight_hh.new_empty(batch_size, self.hidden_size)
                self.noise_hidden = noise.bernoulli_(1.0 - self.p_hidden) / (1.0 - self.p_hidden)
            else:
                self.noise_hidden = None
        else:
            self.noise_in = None
            self.noise_hidden = None

    def forward(self, input, hx):
        return rnn_F.VarFastGRUCell(input, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, self.noise_in, self.noise_hidden)


class VarFastGRU(VarRNNBase):
    """Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::

            \\begin{array}{ll}
            r_t = \\mathrm{sigmoid}(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\\\
            z_t = \\mathrm{sigmoid}(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\\\
            n_t = \\tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\\\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)} \\\\
            \\end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the hidden
    state of the previous layer at time `t` or :math:`input_t` for the first
    layer, and :math:`r_t`, :math:`z_t`, :math:`n_t` are the reset, input,
    and new gates, respectively.

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        nonlinearity: The non-linearity to use ['tanh'|'relu']. Default: 'tanh'
        bias: If False, then the layer does not use bias weights b_ih and b_hh.
            Default: True
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature)
        dropout: (dropout_in, dropout_hidden) tuple.
            If non-zero, introduces a dropout layer on the input and hidden of the each
            RNN layer with dropout rate dropout_in and dropout_hidden, resp.
        bidirectional: If True, becomes a bidirectional RNN. Default: False

    Inputs: input, mask, h_0
        - **input** (seq_len, batch, model_dim): tensor containing the features
          of the input sequence.
          **mask** (seq_len, batch): 0-1 tensor containing the mask of the input sequence.
        - **h_0** (num_layers * num_directions, batch, hidden_size): tensor
          containing the initial hidden state for each element in the batch.

    Outputs: output, h_n
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features (h_k) from the last layer of the RNN,
          for each k.  If a :class:`torch.nn.utils.rnn.PackedSequence` has
          been given as the input, the output will also be a packed sequence.
        - **h_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the hidden state for k=seq_len.
    """

    def __init__(self, *args, **kwargs):
        super(VarFastGRU, self).__init__(VarFastGRUCell, *args, **kwargs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BiLinear,
     lambda: ([], {'left_features': 4, 'right_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChainCRF,
     lambda: ([], {'input_size': 4, 'num_labels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (CharCNN,
     lambda: ([], {'num_layers': 1, 'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_XuezheMax_NeuroNLP2(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

