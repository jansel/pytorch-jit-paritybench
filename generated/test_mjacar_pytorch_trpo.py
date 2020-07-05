import sys
_module = sys.modules[__name__]
del sys
models = _module
run_trpo = _module
trpo_agent = _module
utils = _module
atari_wrapper = _module
math_utils = _module
torch_utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch.nn as nn


import torch.nn.functional as F


import collections


import copy


import torch


from torch.nn.utils.convert_parameters import vector_to_parameters


from torch.nn.utils.convert_parameters import parameters_to_vector


import numpy as np


import torch.optim as optim


class DQNSoftmax(nn.Module):

    def __init__(self, output_size):
        super(DQNSoftmax, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc = nn.Linear(2592, 256)
        self.head = nn.Linear(256, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.fc(out.view(out.size(0), -1)))
        out = self.softmax(self.head(out))
        return out


class DQNRegressor(nn.Module):

    def __init__(self):
        super(DQNRegressor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc = nn.Linear(2592, 256)
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.fc(out.view(out.size(0), -1)))
        out = self.head(out)
        return out


use_cuda = torch.cuda.is_available()


def Variable(tensor, *args, **kwargs):
    if use_cuda:
        return torch.autograd.Variable(tensor, *args, **kwargs).cuda()
    else:
        return torch.autograd.Variable(tensor, *args, **kwargs)


class ValueFunctionWrapper(nn.Module):
    """
  Wrapper around any value function model to add fit and predict functions
  """

    def __init__(self, model, lr):
        super(ValueFunctionWrapper, self).__init__()
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def forward(self, data):
        return self.model.forward(data)

    def fit(self, observations, labels):

        def closure():
            predicted = self.predict(observations)
            loss = self.loss_fn(predicted, labels)
            self.optimizer.zero_grad()
            loss.backward()
            return loss
        old_params = parameters_to_vector(self.model.parameters())
        for lr in (self.lr * 0.5 ** np.arange(10)):
            self.optimizer = optim.LBFGS(self.model.parameters(), lr=lr)
            self.optimizer.step(closure)
            current_params = parameters_to_vector(self.model.parameters())
            if any(np.isnan(current_params.data.cpu().numpy())):
                None
                vector_to_parameters(old_params, self.model.parameters())
            else:
                return

    def predict(self, observations):
        return self.forward(torch.cat([Variable(Tensor(observation)).unsqueeze(0) for observation in observations]))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DQNRegressor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 90, 90])], {}),
     True),
    (DQNSoftmax,
     lambda: ([], {'output_size': 4}),
     lambda: ([torch.rand([4, 1, 90, 90])], {}),
     True),
]

class Test_mjacar_pytorch_trpo(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

