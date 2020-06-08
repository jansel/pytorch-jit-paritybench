import sys
_module = sys.modules[__name__]
del sys
vrae = _module
base = _module
utils = _module
vrae = _module

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


import numpy as np


import torch


from torch import nn


from torch import optim


from torch import distributions


from torch.utils.data import DataLoader


from torch.autograd import Variable


class Encoder(nn.Module):
    """
    Encoder network containing enrolled LSTM/GRU

    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param block: LSTM/GRU block
    """

    def __init__(self, number_of_features, hidden_size, hidden_layer_depth,
        latent_length, dropout, block='LSTM'):
        super(Encoder, self).__init__()
        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size,
                self.hidden_layer_depth, dropout=dropout)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size,
                self.hidden_layer_depth, dropout=dropout)
        else:
            raise NotImplementedError

    def forward(self, x):
        """Forward propagation of encoder. Given input, outputs the last hidden state of encoder

        :param x: input to the encoder, of shape (sequence_length, batch_size, number_of_features)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        """
        _, (h_end, c_end) = self.model(x)
        h_end = h_end[(-1), :, :]
        return h_end


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """

    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()
        self.hidden_size = hidden_size
        self.latent_length = latent_length
        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)
        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        """
        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)
        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean


class Decoder(nn.Module):
    """Converts latent vector into output

    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param block: GRU/LSTM - use the same which you've used in the encoder
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    """

    def __init__(self, sequence_length, batch_size, hidden_size,
        hidden_layer_depth, latent_length, output_size, dtype, block='LSTM'):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype
        if block == 'LSTM':
            self.model = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth)
        elif block == 'GRU':
            self.model = nn.GRU(1, self.hidden_size, self.hidden_layer_depth)
        else:
            raise NotImplementedError
        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)
        self.decoder_inputs = torch.zeros(self.sequence_length, self.
            batch_size, 1, requires_grad=True).type(self.dtype)
        self.c_0 = torch.zeros(self.hidden_layer_depth, self.batch_size,
            self.hidden_size, requires_grad=True).type(self.dtype)
        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent):
        """Converts latent to hidden to output

        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        h_state = self.latent_to_hidden(latent)
        if isinstance(self.model, nn.LSTM):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)]
                )
            decoder_output, _ = self.model(self.decoder_inputs, (h_0, self.c_0)
                )
        elif isinstance(self.model, nn.GRU):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)]
                )
            decoder_output, _ = self.model(self.decoder_inputs, h_0)
        else:
            raise NotImplementedError
        out = self.hidden_to_output(decoder_output)
        return out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_tejaslodaya_timeseries_clustering_vae(_paritybench_base):
    pass

    def test_000(self):
        self._check(Encoder(*[], **{'number_of_features': 4, 'hidden_size': 4, 'hidden_layer_depth': 1, 'latent_length': 4, 'dropout': 0.5}), [torch.rand([4, 4, 4])], {})

    def test_001(self):
        self._check(Lambda(*[], **{'hidden_size': 4, 'latent_length': 4}), [torch.rand([4, 4, 4, 4])], {})
