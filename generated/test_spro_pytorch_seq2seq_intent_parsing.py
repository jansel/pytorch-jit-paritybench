import sys
_module = sys.modules[__name__]
del sys
attn = _module
data = _module
evaluate = _module
model = _module
service = _module
train = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


from torch import optim


import math


USE_CUDA = False


class Attn(nn.Module):

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        attn_energies = Variable(torch.zeros(seq_len))
        if USE_CUDA:
            attn_energies = attn_energies
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, input):
        seq_len = input.size(0)
        batch_size = input.size(1)
        embedded = self.embedding(input.view(seq_len * batch_size, -1))
        output = embedded.view(seq_len, batch_size, -1)
        output, hidden = self.gru(output)
        return output, hidden


class AttnDecoderRNN(nn.Module):

    def __init__(self, attn_model, hidden_size, output_size, n_layers=1,
        dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p
            )
        self.out = nn.Linear(hidden_size * 2, output_size)
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        word_embedded = self.embedding(word_input).view(1, 1, -1)
        rnn_output, hidden = self.gru(word_embedded, last_hidden)
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        return output, hidden, attn_weights


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_spro_pytorch_seq2seq_intent_parsing(_paritybench_base):
    pass
