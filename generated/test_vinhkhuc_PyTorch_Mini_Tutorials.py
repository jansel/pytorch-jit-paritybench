import sys
_module = sys.modules[__name__]
del sys
data_util = _module

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


from torch.autograd import Variable


from torch import optim


import numpy as np


from torch import nn


class ConvNet(torch.nn.Module):

    def __init__(self, output_dim):
        super(ConvNet, self).__init__()
        self.conv = torch.nn.Sequential()
        self.conv.add_module('conv_1', torch.nn.Conv2d(1, 10, kernel_size=5))
        self.conv.add_module('maxpool_1', torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module('relu_1', torch.nn.ReLU())
        self.conv.add_module('conv_2', torch.nn.Conv2d(10, 20, kernel_size=5))
        self.conv.add_module('dropout_2', torch.nn.Dropout())
        self.conv.add_module('maxpool_2', torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module('relu_2', torch.nn.ReLU())
        self.fc = torch.nn.Sequential()
        self.fc.add_module('fc1', torch.nn.Linear(320, 50))
        self.fc.add_module('relu_3', torch.nn.ReLU())
        self.fc.add_module('dropout_3', torch.nn.Dropout())
        self.fc.add_module('fc2', torch.nn.Linear(50, output_dim))

    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(-1, 320)
        return self.fc.forward(x)


class LSTMNet(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        batch_size = x.size()[1]
        h0 = Variable(torch.zeros([1, batch_size, self.hidden_dim]), requires_grad=False)
        c0 = Variable(torch.zeros([1, batch_size, self.hidden_dim]), requires_grad=False)
        fx, _ = self.lstm.forward(x, (h0, c0))
        return self.linear.forward(fx[-1])


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LSTMNet,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_vinhkhuc_PyTorch_Mini_Tutorials(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

