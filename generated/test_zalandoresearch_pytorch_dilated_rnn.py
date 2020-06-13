import sys
_module = sys.modules[__name__]
del sys
master = _module
char_rnn_test = _module
model = _module
utils = _module
classify = _module
copymem_test = _module
model = _module
drnn = _module
lm = _module
tests = _module

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


import math


import torch.nn as nn


import torch.optim as optim


import warnings


from torch import nn


import torch


import torch.autograd as autograd


import torch.utils.data as Data


import numpy as np


from torch.autograd import Variable


class DRNN_Char(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size,
        dropout=0.2, emb_dropout=0.2):
        super(DRNN_Char, self).__init__()
        self.encoder = nn.Embedding(output_size, input_size)
        self.drnn = DRNN(cell_type='QRNN', dropout=dropout, n_hidden=
            hidden_size, n_input=input_size, n_layers=num_layers,
            batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.drop = nn.Dropout(emb_dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        emb = self.drop(self.encoder(x))
        y, _ = self.drnn(emb)
        o = self.decoder(y)
        return o.contiguous()


class Classifier(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_layers, n_classes, cell_type='GRU'
        ):
        super(Classifier, self).__init__()
        self.drnn = drnn.DRNN(n_inputs, n_hidden, n_layers, dropout=0,
            cell_type=cell_type)
        self.linear = nn.Linear(n_hidden, n_classes)

    def forward(self, inputs):
        layer_outputs, _ = self.drnn(inputs)
        pred = self.linear(layer_outputs[-1])
        return pred


class DRNN_Copy(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout,
        output_size):
        super(DRNN_Copy, self).__init__()
        self.drnn = DRNN(cell_type='GRU', dropout=dropout, n_hidden=
            hidden_size, n_input=input_size, n_layers=num_layers,
            batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1, _ = self.drnn(x)
        return self.linear(y1)


use_cuda = torch.cuda.is_available()


class DRNN(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers, dropout=0, cell_type=
        'GRU', batch_first=False):
        super(DRNN, self).__init__()
        self.dilations = [(2 ** i) for i in range(n_layers)]
        self.cell_type = cell_type
        self.batch_first = batch_first
        layers = []
        if self.cell_type == 'GRU':
            cell = nn.GRU
        elif self.cell_type == 'RNN':
            cell = nn.RNN
        elif self.cell_type == 'LSTM':
            cell = nn.LSTM
        else:
            raise NotImplementedError
        for i in range(n_layers):
            if i == 0:
                c = cell(n_input, n_hidden, dropout=dropout)
            else:
                c = cell(n_hidden, n_hidden, dropout=dropout)
            layers.append(c)
        self.cells = nn.Sequential(*layers)

    def forward(self, inputs, hidden=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inputs, _ = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation,
                    hidden[i])
            outputs.append(inputs[-dilation:])
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        return inputs, outputs

    def drnn_layer(self, cell, inputs, rate, hidden=None):
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size
        inputs, _ = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)
        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell,
                batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell,
                batch_size, rate, hidden_size, hidden=hidden)
        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)
        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate,
        hidden_size, hidden=None):
        if hidden is None:
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                hidden = c.unsqueeze(0), m.unsqueeze(0)
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size
                    ).unsqueeze(0)
        dilated_outputs, hidden = cell(dilated_inputs, hidden)
        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate
        blocks = [dilated_outputs[:, i * batchsize:(i + 1) * batchsize, :] for
            i in range(rate)]
        interleaved = torch.stack(blocks).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
            batchsize, dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):
        is_even = n_steps % rate == 0
        if not is_even:
            dilated_steps = n_steps // rate + 1
            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                inputs.size(1), inputs.size(2))
            if use_cuda:
                zeros_ = zeros_
            inputs = torch.cat((inputs, zeros_))
        else:
            dilated_steps = n_steps // rate
        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(
            rate)], 1)
        return dilated_inputs

    def init_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(batch_size, hidden_dim)
        if use_cuda:
            hidden = hidden
        if self.cell_type == 'LSTM':
            memory = torch.zeros(batch_size, hidden_dim)
            if use_cuda:
                memory = memory
            return hidden, memory
        else:
            return hidden


class RNNModel(nn.Module):

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
        tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=
                dropout)
        elif rnn_type == 'DRNN':
            self.rnn = drnn.DRNN(ninp, nhid, nlayers, 0, 'GRU')
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    'When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        try:
            self.init_weights()
        except AttributeError:
            pass
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1),
            output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)
            ), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()
                ), Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_zalandoresearch_pytorch_dilated_rnn(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Classifier(*[], **{'n_inputs': 4, 'n_hidden': 4, 'n_layers': 1, 'n_classes': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(DRNN_Copy(*[], **{'input_size': 4, 'hidden_size': 4, 'num_layers': 1, 'dropout': 0.5, 'output_size': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(DRNN(*[], **{'n_input': 4, 'n_hidden': 4, 'n_layers': 1}), [torch.rand([4, 4, 4])], {})

