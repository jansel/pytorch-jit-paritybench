import sys
_module = sys.modules[__name__]
del sys
model = _module
tsp = _module
vrp = _module
trainer = _module

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


import torch.nn as nn


import torch.nn.functional as F


import time


import numpy as np


import torch.optim as optim


from torch.utils.data import DataLoader


class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), device=
            device, requires_grad=True))
        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size),
            device=device, requires_grad=True))

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):
        batch_size, hidden_size, _ = static_hidden.size()
        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)
        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)
        return attns


class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), device=
            device, requires_grad=True))
        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
            device=device, requires_grad=True))
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first
            =True, dropout=dropout if num_layers > 1 else 0)
        self.encoder_attn = Attention(hidden_size)
        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):
        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            last_hh = self.drop_hh(last_hh)
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out)
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)
        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)
        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)
        return probs, last_hh


class DRL4TSP(nn.Module):
    """Defines the main Encoder, Decoder, and Pointer combinatorial models.

    Parameters
    ----------
    static_size: int
        Defines how many features are in the static elements of the model
        (e.g. 2 for (x, y) coordinates)
    dynamic_size: int > 1
        Defines how many features are in the dynamic elements of the model
        (e.g. 2 for the VRP which has (load, demand) attributes. The TSP doesn't
        have dynamic elements, but to ensure compatility with other optimization
        problems, assume we just pass in a vector of zeros.
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
    update_fn: function or None
        If provided, this method is used to calculate how the input dynamic
        elements are updated, and is called after each 'point' to the input element.
    mask_fn: function or None
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidlines to the algorithm. If no mask
        is provided, we terminate the search after a fixed number of iterations
        to avoid tours that stretch forever
    num_layers: int
        Specifies the number of hidden layers to use in the decoder RNN
    dropout: float
        Defines the dropout rate for the decoder
    """

    def __init__(self, static_size, dynamic_size, hidden_size, update_fn=
        None, mask_fn=None, num_layers=1, dropout=0.0):
        super(DRL4TSP, self).__init__()
        if dynamic_size < 1:
            raise ValueError(
                ':param dynamic_size: must be > 0, even if the problem has no dynamic elements'
                )
        self.update_fn = update_fn
        self.mask_fn = mask_fn
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Encoder(static_size, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
        self.x0 = torch.zeros((1, static_size, 1), requires_grad=True,
            device=device)

    def forward(self, static, dynamic, decoder_input=None, last_hh=None):
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
        dynamic: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the VRP, this can be
            things like the (load, demand) of each city. If there are no dynamic
            elements, this can be set to None
        decoder_input: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
        """
        batch_size, input_size, sequence_size = static.size()
        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, -1, -1)
        mask = torch.ones(batch_size, sequence_size, device=device)
        tour_idx, tour_logp = [], []
        max_steps = sequence_size if self.mask_fn is None else 1000
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)
        for _ in range(max_steps):
            if not mask.byte().any():
                break
            decoder_hidden = self.decoder(decoder_input)
            probs, last_hh = self.pointer(static_hidden, dynamic_hidden,
                decoder_hidden, last_hh)
            probs = F.softmax(probs + mask.log(), dim=1)
            if self.training:
                m = torch.distributions.Categorical(probs)
                ptr = m.sample()
                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte(
                    ).all():
                    ptr = m.sample()
                logp = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1)
                logp = prob.log()
            if self.update_fn is not None:
                dynamic = self.update_fn(dynamic, ptr.data)
                dynamic_hidden = self.dynamic_encoder(dynamic)
                is_done = dynamic[:, (1)].sum(1).eq(0).float()
                logp = logp * (1.0 - is_done)
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, dynamic, ptr.data).detach()
            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))
            decoder_input = torch.gather(static, 2, ptr.view(-1, 1, 1).
                expand(-1, input_size, 1)).detach()
        tour_idx = torch.cat(tour_idx, dim=1)
        tour_logp = torch.cat(tour_logp, dim=1)
        return tour_idx, tour_logp


class StateCritic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)
        hidden = torch.cat((static_hidden, dynamic_hidden), 1)
        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output


class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):
        output = F.relu(self.fc1(input.unsqueeze(1)))
        output = F.relu(self.fc2(output)).squeeze(2)
        output = self.fc3(output).sum(dim=2)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_mveres01_pytorch_drl4vrp(_paritybench_base):
    pass
    def test_000(self):
        self._check(Attention(*[], **{'hidden_size': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])], {})

    def test_001(self):
        self._check(Critic(*[], **{'hidden_size': 4}), [torch.rand([4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(DRL4TSP(*[], **{'static_size': 4, 'dynamic_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})

    def test_003(self):
        self._check(Encoder(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 64])], {})

    def test_004(self):
        self._check(StateCritic(*[], **{'static_size': 4, 'dynamic_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 64]), torch.rand([4, 4, 64])], {})

