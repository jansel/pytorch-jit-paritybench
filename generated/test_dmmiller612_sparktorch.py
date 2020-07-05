import sys
_module = sys.modules[__name__]
del sys
cnn_network = _module
lazy_load_cnn = _module
simple_cnn = _module
simple_dnn = _module
setup = _module
sparktorch = _module
distributed = _module
early_stopper = _module
hogwild = _module
inference = _module
pipeline_util = _module
rw_lock = _module
server = _module
tests = _module
simple_net = _module
test_sparktorch = _module
torch_distributed = _module
util = _module

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


import torch


import numpy as np


from torch.optim.optimizer import Optimizer


from typing import Any


from typing import Dict


from typing import List


from typing import Type


from typing import Tuple


import collections


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(3872, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.bottle = nn.Linear(5, 2)
        self.fc2 = nn.Linear(2, 5)
        self.out = nn.Linear(5, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bottle(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        return x


class ClassificationNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class NetworkWithParameters(nn.Module):

    def __init__(self, param):
        super().__init__()
        self.fc1 = nn.Linear(10, param)
        self.fc2 = nn.Linear(param, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AutoEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([10, 10])], {}),
     True),
    (ClassificationNet,
     lambda: ([], {}),
     lambda: ([torch.rand([10, 10])], {}),
     True),
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([10, 10])], {}),
     True),
    (NetworkWithParameters,
     lambda: ([], {'param': 4}),
     lambda: ([torch.rand([10, 10])], {}),
     True),
]

class Test_dmmiller612_sparktorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

