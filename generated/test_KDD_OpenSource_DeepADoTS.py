import sys
_module = sys.modules[__name__]
del sys
experiments = _module
main = _module
setup = _module
src = _module
algorithms = _module
algorithm_utils = _module
autoencoder = _module
dagmm = _module
donut = _module
lstm_ad = _module
lstm_enc_dec_axl = _module
rnn_ebm = _module
data_loader = _module
datasets = _module
dataset = _module
kdd_cup = _module
multivariate_anomaly_function = _module
real_datasets = _module
synthetic_data_generator = _module
synthetic_dataset = _module
synthetic_multivariate_dataset = _module
evaluation = _module
config = _module
evaluator = _module
plotter = _module
tests = _module
test_initialization = _module

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


import logging


import numpy as np


import torch


import torch.nn as nn


from scipy.stats import multivariate_normal


from torch.utils.data import DataLoader


from torch.utils.data.sampler import SubsetRandomSampler


import torch.nn.functional as F


from torch.autograd import Variable


class LSTMSequence(torch.nn.Module):

    def __init__(self, d, batch_size: int, len_in=1, len_out=10):
        super().__init__()
        self.d = d
        self.batch_size = batch_size
        self.len_in = len_in
        self.len_out = len_out
        self.hidden_size1 = 32
        self.hidden_size2 = 32
        self.lstm1 = torch.nn.LSTMCell(d * len_in, self.hidden_size1)
        self.lstm2 = torch.nn.LSTMCell(self.hidden_size1, self.hidden_size2)
        self.linear = torch.nn.Linear(self.hidden_size2, d * len_out)
        self.register_buffer('h_t', torch.zeros(self.batch_size, self.
            hidden_size1))
        self.register_buffer('c_t', torch.zeros(self.batch_size, self.
            hidden_size1))
        self.register_buffer('h_t2', torch.zeros(self.batch_size, self.
            hidden_size1))
        self.register_buffer('c_t2', torch.zeros(self.batch_size, self.
            hidden_size1))

    def forward(self, input):
        outputs = []
        h_t = Variable(self.h_t.double(), requires_grad=False)
        c_t = Variable(self.c_t.double(), requires_grad=False)
        h_t2 = Variable(self.h_t2.double(), requires_grad=False)
        c_t2 = Variable(self.c_t2.double(), requires_grad=False)
        for input_t in input.chunk(input.size(1), dim=1):
            h_t, c_t = self.lstm1(input_t.squeeze(dim=1), (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze()
        return outputs.view(input.size(0), input.size(1), self.d, self.len_out)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_KDD_OpenSource_DeepADoTS(_paritybench_base):
    pass
