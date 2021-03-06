import sys
_module = sys.modules[__name__]
del sys
master = _module
algorithms = _module
maddpg = _module
evaluate = _module
main = _module
utils = _module
agents = _module
buffer = _module
env_wrappers = _module
make_env = _module
misc = _module
networks = _module
noise = _module

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


import torch.nn.functional as F


import time


import numpy as np


from torch.autograd import Variable


from torch import Tensor


from torch.optim import Adam


import torch.distributed as dist


import torch.nn as nn


class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """

    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu, constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()
        if norm_in:
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            self.fc3.weight.data.uniform_(-0.003, 0.003)
            self.out_fn = F.tanh
        else:
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MLPNetwork,
     lambda: ([], {'input_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_shariqiqbal2810_maddpg_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

