import sys
_module = sys.modules[__name__]
del sys
GenCaptcha = _module
ctcmain = _module
ctcmodel = _module
data_utils_torch = _module
main = _module
model = _module
utils = _module

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


import matplotlib


import matplotlib.pyplot as plt


import torch


import torch.nn as nn


from torch.autograd import Variable


import torch.nn.functional as F


from torch import optim


import numpy as np


from torch.utils.data import TensorDataset


from torch.utils.data import DataLoader


from torch.nn import functional


class ResBlock(nn.Module):

    def __init__(self, in_c, out_c, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False)
        self.in_c = in_c
        self.out_c = out_c

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.in_c != self.out_c:
            return out + self.downsample(x)
        else:
            return out + x


class CTCModel(nn.Module):

    def __init__(self, output_size, rnn_hidden_size=128, num_rnn_layers=1, dropout=0):
        super(CTCModel, self).__init__()
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.output_size = output_size
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU(), nn.Conv2d(32, 32, kernel_size=(3, 4), stride=(3, 2)), nn.BatchNorm2d(32), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU(), nn.Conv2d(32, 32, kernel_size=(4, 3), stride=(4, 2)), nn.BatchNorm2d(32), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU(), nn.Conv2d(32, 32, kernel_size=(4, 2), stride=(1, 1)), nn.BatchNorm2d(32), nn.ReLU())
        self.gru = nn.GRU(32, rnn_hidden_size, num_rnn_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.linear = nn.Linear(rnn_hidden_size * 2, output_size)

    def forward(self, x, hidden):
        h0 = hidden
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out).squeeze()
        out = out.transpose(1, 2)
        out, hidden = self.gru(out, h0)
        out = self.linear(out)
        return out

    def initHidden(self, batch_size, use_cuda=False):
        h0 = Variable(torch.zeros(self.num_rnn_layers * 2, batch_size, self.rnn_hidden_size))
        if use_cuda:
            return h0
        else:
            return h0


class Encoder(nn.Module):

    def __init__(self, ENC_TYPE, num_rnn_layers=1, rnn_hidden_size=128, dropout=0):
        super(Encoder, self).__init__()
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.ENC_TYPE = ENC_TYPE
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=(3, 4), stride=(3, 2)), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(dropout))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(4, 3), stride=(4, 2)), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(dropout))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(4, 2), stride=(1, 1)), nn.BatchNorm2d(128), nn.ReLU())
        self.gru = nn.GRU(128, rnn_hidden_size, num_rnn_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(128, rnn_hidden_size * num_rnn_layers)
        self.layer4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.layer5 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.layer6 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x, hidden):
        if self.ENC_TYPE == 'CNNRNN':
            h0 = hidden
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out).squeeze()
            out = out.transpose(1, 2)
            out, hidden = self.gru(out, h0)
            return out, hidden
        else:
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out).squeeze()
            for layer in [self.layer4, self.layer5, self.layer6]:
                input = out
                out = layer(out)
                out = out[:, :128] * nn.functional.sigmoid(out[:, 128:])
                out = out + input
            out = out.transpose(1, 2)
            hidden = torch.nn.functional.tanh(self.linear(out.mean(dim=1))).view(self.num_rnn_layers, -1, self.rnn_hidden_size)
            return out, hidden

    def initHidden(self, batch_size, use_cuda=False):
        h0 = Variable(torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_size))
        if use_cuda:
            return h0
        else:
            return h0


class Attn(nn.Module):

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size, bias=False)
        elif self.method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.tanh = nn.Tanh()
            self.attn_linear = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        :param hidden: decode hidden state, (batch_size , N)
        :param encoder_outputs: encoder's all states, (batch_size,T,N)
        :return: weithed_context :(batch_size,N), alpha:(batch_size,T)
        """
        hidden_expanded = hidden.unsqueeze(2)
        if self.method == 'dot':
            energy = torch.bmm(encoder_outputs, hidden_expanded).squeeze(2)
        elif self.method == 'general':
            energy = self.attn(encoder_outputs)
            energy = torch.bmm(energy, hidden_expanded).squeeze(2)
        elif self.method == 'concat':
            hidden_expanded = hidden.unsqueeze(1).expand_as(encoder_outputs)
            energy = self.attn(torch.cat((hidden_expanded, encoder_outputs), 2))
            energy = self.attn_linear(self.tanh(energy)).squeeze(2)
        alpha = nn.functional.softmax(energy)
        weighted_context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return weighted_context, alpha


class RNNAttnDecoder(nn.Module):

    def __init__(self, attn_model, input_vocab_size, hidden_size, output_size, num_rnn_layers=1, dropout=0.0):
        super(RNNAttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_vocab_size = input_vocab_size
        self.output_size = output_size
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
            self.gru = nn.GRU(input_vocab_size + hidden_size, hidden_size, num_rnn_layers, batch_first=True, dropout=dropout)
        else:
            self.attn = None
            self.gru = nn.GRU(input_vocab_size, hidden_size, num_rnn_layers, batch_first=True, dropout=dropout)
        self.wc = nn.Linear(2 * hidden_size, hidden_size)
        self.ws = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(input_vocab_size, input_vocab_size)
        fix_embedding = torch.from_numpy(np.eye(input_vocab_size, input_vocab_size).astype(np.float32))
        self.embedding.weight = nn.Parameter(fix_embedding)
        self.embedding.weight.requires_grad = False

    def forward(self, input, last_ht, last_hidden, encoder_outputs):
        """
        :se
        :param input: (batch_size,)
        :param last_ht: (obatch_size,hidden_size)
        :param last_hidden: (batch_size,hidden_size)
        :param encoder_outputs: (batch_size,T,hidden_size)
        """
        if self.attn is None:
            embed_input = self.embedding(input.unsqueeze(1))
            output, hidden = self.gru(embed_input, last_hidden)
            output = self.ws(output.squeeze())
            return output, last_ht, hidden, None
        else:
            embed_input = self.embedding(input)
            rnn_input = torch.cat((embed_input, last_ht), 1)
            output, hidden = self.gru(rnn_input.unsqueeze(1), last_hidden)
            output = output.squeeze()
            weighted_context, alpha = self.attn(output, encoder_outputs)
            ht = self.tanh(self.wc(torch.cat((output, weighted_context), 1)))
            output = self.ws(ht)
            return output, ht, hidden, alpha


class RNNAttnDecoder2(nn.Module):

    def __init__(self, attn_model, input_vocab_size, hidden_size, output_size, num_rnn_layers=1, dropout=0.0):
        super(RNNAttnDecoder2, self).__init__()
        self.hidden_size = hidden_size
        self.input_vocab_size = input_vocab_size
        self.output_size = output_size
        self.attn = Attn(attn_model, hidden_size)
        self.gru = nn.GRU(input_vocab_size + hidden_size, hidden_size, num_rnn_layers, batch_first=True, dropout=dropout)
        self.embedding = nn.Embedding(input_vocab_size, input_vocab_size)
        fix_embedding = torch.from_numpy(np.eye(input_vocab_size, input_vocab_size).astype(np.float32))
        self.embedding.weight = nn.Parameter(fix_embedding)
        self.embedding.weight.requires_grad = False
        self.wc = nn.Linear(hidden_size + input_vocab_size, hidden_size)
        self.ws = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_ht, last_hidden, encoder_outputs):
        embed_input = self.embedding(input)
        attn_input = self.wc(torch.cat((embed_input, last_hidden[-1]), 1))
        weighted_context, alpha = self.attn(attn_input, encoder_outputs)
        rnn_input = torch.cat((embed_input, weighted_context), 1)
        output, hidden = self.gru(rnn_input.unsqueeze(1), last_hidden)
        output = output.squeeze()
        output = self.ws(torch.cat((output, nn.functional.tanh(weighted_context)), 1))
        return output, last_ht, hidden, alpha


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ResBlock,
     lambda: ([], {'in_c': 4, 'out_c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_airaria_CaptchaRecognition(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

