import sys
_module = sys.modules[__name__]
del sys
generate = _module
helpers = _module
model = _module
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


from torch.autograd import Variable


class CharRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, model='gru',
        n_layers=1):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == 'gru':
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def forward2(self, input, hidden):
        encoded = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == 'lstm':
            return Variable(torch.zeros(self.n_layers, batch_size, self.
                hidden_size)), Variable(torch.zeros(self.n_layers,
                batch_size, self.hidden_size))
        return Variable(torch.zeros(self.n_layers, batch_size, self.
            hidden_size))


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_spro_char_rnn_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(CharRNN(*[], **{'input_size': 4, 'hidden_size': 4, 'output_size': 4}), [torch.zeros([4], dtype=torch.int64), torch.rand([1, 4, 4])], {})

