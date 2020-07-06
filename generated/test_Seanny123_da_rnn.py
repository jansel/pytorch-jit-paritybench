import sys
_module = sys.modules[__name__]
del sys
constants = _module
custom_types = _module
main = _module
main_predict = _module
modules = _module
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


import torch


import typing


from typing import Tuple


from torch import nn


from torch import optim


from sklearn.preprocessing import StandardScaler


import numpy as np


from torch.autograd import Variable


from torch.nn import functional as tf


import logging


def init_hidden(x, hidden_size: int):
    """
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    return Variable(torch.zeros(1, x.size(0), hidden_size))


class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, T: int):
        """
        input size: number of underlying factors (81)
        T: number of time steps (10)
        hidden_size: dimension of the hidden state
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + T - 1, out_features=1)

    def forward(self, input_data):
        input_weighted = Variable(torch.zeros(input_data.size(0), self.T - 1, self.input_size))
        input_encoded = Variable(torch.zeros(input_data.size(0), self.T - 1, self.hidden_size))
        hidden = init_hidden(input_data, self.hidden_size)
        cell = init_hidden(input_data, self.hidden_size)
        for t in range(self.T - 1):
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2), cell.repeat(self.input_size, 1, 1).permute(1, 0, 2), input_data.permute(0, 2, 1)), dim=2)
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T - 1))
            attn_weights = tf.softmax(x.view(-1, self.input_size), dim=1)
            weighted_input = torch.mul(attn_weights, input_data[:, (t), :])
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            input_weighted[:, (t), :] = weighted_input
            input_encoded[:, (t), :] = hidden
        return input_weighted, input_encoded


class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, out_feats=1):
        super(Decoder, self).__init__()
        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size), nn.Tanh(), nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)
        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        hidden = init_hidden(input_encoded, self.decoder_hidden_size)
        cell = init_hidden(input_encoded, self.decoder_hidden_size)
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))
        for t in range(self.T - 1):
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2), cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2), input_encoded), dim=2)
            x = tf.softmax(self.attn_layer(x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)).view(-1, self.T - 1), dim=1)
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, (0), :]
            y_tilde = self.fc(torch.cat((context, y_history[:, (t)]), dim=1))
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0]
            cell = lstm_output[1]
        return self.fc_final(torch.cat((hidden[0], context), dim=1))

