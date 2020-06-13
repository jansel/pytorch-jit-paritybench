import sys
_module = sys.modules[__name__]
del sys
dump_loglikes = _module
latgen = _module
train_ce = _module
train_chain = _module
train_se = _module
train_se2 = _module
train_transformer_ce = _module
train_transformer_se = _module
data = _module
dataloader = _module
sr_dataset = _module
decode = _module
utt_wer = _module
models = _module
lstm = _module
lstm_libcss = _module
transformer = _module
ops = _module
ops = _module
reader = _module
preprocess = _module
stream = _module
zip_io = _module
simulation = _module
_distorter = _module
_geometry = _module
_iso_noise_simulator = _module
_mixer = _module
_rirgen = _module
_sampling = _module
config = _module
freq_analysis = _module
mask = _module
overlap = _module
utils = _module

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


import torch as th


import torch.nn as nn


import math


import torch.nn.functional as F


from torch.autograd import Function


class LSTMAM(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers,
        dropout, bidirectional):
        super(LSTMStack, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        if bidirectional:
            self.output_layer = nn.Linear(hidden_size * 2, output_size)
        else:
            self.output_layer = nn.Linear(hidden_size, output_size)
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.
            hidden_size, num_layers=self.num_layers, batch_first=True,
            dropout=self.dropout, bidirectional=self.bidirectional)

    def forward(self, data):
        self.lstm.flatten_parameters()
        output, (h, c) = self.lstm(data)
        output = self_output_layer(output)
        return output


class LSTMStack(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout,
        bidirectional):
        super(LSTMStack, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.
            hidden_size, num_layers=self.num_layers, batch_first=True,
            dropout=self.dropout, bidirectional=self.bidirectional)

    def forward(self, data):
        self.lstm.flatten_parameters()
        output, (h, c) = self.lstm(data)
        return output, (h, c)


class NnetAM(nn.Module):

    def __init__(self, nnet, hidden_size, output_size):
        super(NnetAM, self).__init__()
        self.nnet = nnet
        self.output_size = output_size
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, data):
        nnet_output, (h, c) = self.nnet(data)
        output = self.output_layer(nnet_output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, dim_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = th.zeros(max_len, dim_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, dim_model, 2).float() * (-math.log(
            10000.0) / dim_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayerWithConv1d(nn.Module):
    """
      Input and output shape: seqlen x batch_size x dim
    """

    def __init__(self, dim_model, nheads, dim_feedforward, dropout,
        kernel_size, stride):
        super(TransformerEncoderLayerWithConv1d, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(dim_model, nheads,
            dim_feedforward, dropout)
        self.conv1d = nn.Conv1d(dim_model, dim_model, kernel_size, stride=
            stride, padding=1)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = self.encoder_layer(src, src_mask, src_key_padding_mask)
        output = F.relu(self.conv1d(output.permute(1, 2, 0)))
        return output.permute(2, 0, 1)


class TransformerAM(nn.Module):

    def __init__(self, dim_feat, dim_model, nheads, dim_feedforward,
        nlayers, dropout, output_size, kernel_size=3, stride=1):
        super(TransformerAM, self).__init__()
        self.pos_encoder = PositionalEncoding(dim_model, dropout)
        self.input_layer = nn.Linear(dim_feat, dim_model)
        self.output_layer = nn.Linear(dim_model, output_size)
        encoder_norm = nn.LayerNorm(dim_model)
        encoder_layer = TransformerEncoderLayerWithConv1d(dim_model, nheads,
            dim_feedforward, dropout, kernel_size, stride)
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers,
            norm=encoder_norm)

    def forward(self, data, src_mask=None, src_key_padding_mask=None):
        input = self.input_layer(data)
        output = self.transformer(input, mask=src_mask,
            src_key_padding_mask=src_key_padding_mask)
        output = self.output_layer(output)
        return output


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jzlianglu_pykaldi2(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(PositionalEncoding(*[], **{'dim_model': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(TransformerAM(*[], **{'dim_feat': 4, 'dim_model': 4, 'nheads': 4, 'dim_feedforward': 4, 'nlayers': 1, 'dropout': 0.5, 'output_size': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(TransformerEncoderLayerWithConv1d(*[], **{'dim_model': 4, 'nheads': 4, 'dim_feedforward': 4, 'dropout': 0.5, 'kernel_size': 4, 'stride': 1}), [torch.rand([4, 4, 4])], {})

