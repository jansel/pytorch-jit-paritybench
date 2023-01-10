import sys
_module = sys.modules[__name__]
del sys
config = _module
dataset = _module
model = _module
trainer = _module

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


import torch


from torch import nn


from torch.autograd import Variable


import torch.nn.functional as F


import matplotlib


import matplotlib.pyplot as plt


import numpy as np


from torch import optim


class AttnEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, time_step):
        super(AttnEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = time_step
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=self.T)
        self.attn2 = nn.Linear(in_features=self.T, out_features=self.T)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=self.T, out_features=1)

    def forward(self, driving_x):
        batch_size = driving_x.size(0)
        code = self.init_variable(batch_size, self.T, self.hidden_size)
        h = self.init_variable(1, batch_size, self.hidden_size)
        s = self.init_variable(1, batch_size, self.hidden_size)
        for t in range(self.T):
            x = torch.cat((self.embedding_hidden(h), self.embedding_hidden(s)), 2)
            z1 = self.attn1(x)
            z2 = self.attn2(driving_x.permute(0, 2, 1))
            x = z1 + z2
            z3 = self.attn3(self.tanh(x))
            if batch_size > 1:
                attn_w = F.softmax(z3.view(batch_size, self.input_size), dim=1)
            else:
                attn_w = self.init_variable(batch_size, self.input_size) + 1
            weighted_x = torch.mul(attn_w, driving_x[:, t, :])
            _, states = self.lstm(weighted_x.unsqueeze(0), (h, s))
            h = states[0]
            s = states[1]
            code[:, t, :] = h
        return code

    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor
        return Variable(zero_tensor)

    def embedding_hidden(self, x):
        return x.repeat(self.input_size, 1, 1).permute(1, 0, 2)


class AttnDecoder(nn.Module):

    def __init__(self, code_hidden_size, hidden_size, time_step):
        super(AttnDecoder, self).__init__()
        self.code_hidden_size = code_hidden_size
        self.hidden_size = hidden_size
        self.T = time_step
        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=code_hidden_size)
        self.attn2 = nn.Linear(in_features=code_hidden_size, out_features=code_hidden_size)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=code_hidden_size, out_features=1)
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size)
        self.tilde = nn.Linear(in_features=self.code_hidden_size + 1, out_features=1)
        self.fc1 = nn.Linear(in_features=code_hidden_size + hidden_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, h, y_seq):
        batch_size = h.size(0)
        d = self.init_variable(1, batch_size, self.hidden_size)
        s = self.init_variable(1, batch_size, self.hidden_size)
        ct = self.init_variable(batch_size, self.hidden_size)
        for t in range(self.T):
            x = torch.cat((self.embedding_hidden(d), self.embedding_hidden(s)), 2)
            z1 = self.attn1(x)
            z2 = self.attn2(h)
            x = z1 + z2
            z3 = self.attn3(self.tanh(x))
            if batch_size > 1:
                beta_t = F.softmax(z3.view(batch_size, -1), dim=1)
            else:
                beta_t = self.init_variable(batch_size, self.code_hidden_size) + 1
            ct = torch.bmm(beta_t.unsqueeze(1), h).squeeze(1)
            if t < self.T - 1:
                yc = torch.cat((y_seq[:, t].unsqueeze(1), ct), dim=1)
                y_tilde = self.tilde(yc)
                _, states = self.lstm(y_tilde.unsqueeze(0), (d, s))
                d = states[0]
                s = states[1]
        y_res = self.fc2(self.fc1(torch.cat((d.squeeze(0), ct), dim=1)))
        return y_res

    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor
        return Variable(zero_tensor)

    def embedding_hidden(self, x):
        return x.repeat(self.T, 1, 1).permute(1, 0, 2)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttnDecoder,
     lambda: ([], {'code_hidden_size': 4, 'hidden_size': 4, 'time_step': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (AttnEncoder,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'time_step': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_ysn2233_attentioned_dual_stage_stock_prediction(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

