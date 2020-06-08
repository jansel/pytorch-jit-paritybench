import sys
_module = sys.modules[__name__]
del sys
conv_lstm = _module

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


import torch.nn as nn


from torch.autograd import Variable


import torch


class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size
      
    """

    def __init__(self, shape, input_chans, filter_size, num_features):
        super(CLSTM_cell, self).__init__()
        self.shape = shape
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) / 2
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 *
            self.num_features, self.filter_size, 1, self.padding)

    def forward(self, input, hidden_state):
        hidden, c = hidden_state
        combined = torch.cat((input, hidden), 1)
        A = self.conv(combined)
        ai, af, ao, ag = torch.split(A, self.num_features, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)
        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.num_features, self.
            shape[0], self.shape[1])), Variable(torch.zeros(batch_size,
            self.num_features, self.shape[0], self.shape[1]))


class CLSTM(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size
      
    """

    def __init__(self, shape, input_chans, filter_size, num_features,
        num_layers):
        super(CLSTM, self).__init__()
        self.shape = shape
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.num_layers = num_layers
        cell_list = []
        cell_list.append(CLSTM_cell(self.shape, self.input_chans, self.
            filter_size, self.num_features))
        for idcell in range(1, self.num_layers):
            cell_list.append(CLSTM_cell(self.shape, self.num_features, self
                .filter_size, self.num_features))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input, hidden_state):
        """
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            input is the tensor of shape seq_len,Batch,Chans,H,W

        """
        current_input = input.transpose(0, 1)
        next_hidden = []
        seq_len = current_input.size(0)
        for idlayer in range(self.num_layers):
            hidden_c = hidden_state[idlayer]
            all_output = []
            output_inner = []
            for t in range(seq_len):
                hidden_c = self.cell_list[idlayer](current_input[t, ...],
                    hidden_c)
                output_inner.append(hidden_c[0])
            next_hidden.append(hidden_c)
            current_input = torch.cat(output_inner, 0).view(current_input.
                size(0), *output_inner[0].size())
        return next_hidden, current_input

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_rogertrullo_pytorch_convlstm(_paritybench_base):
    pass
