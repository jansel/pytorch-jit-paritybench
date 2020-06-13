import sys
_module = sys.modules[__name__]
del sys
AwA1_RN = _module
AwA2_RN = _module
CUB_RN = _module

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


import torch.nn.functional as F


from torch.autograd import Variable


from torch.optim.lr_scheduler import StepLR


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


import numpy as np


import scipy.io as sio


import math


import random


class AttributeNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size, output_size):
        super(AttributeNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


class AttributeNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size, output_size):
        super(AttributeNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


class AttributeNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size, output_size):
        super(AttributeNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_lzrobots_LearningToCompare_ZSL(_paritybench_base):
    pass
    def test_000(self):
        self._check(AttributeNetwork(*[], **{'input_size': 4, 'hidden_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(RelationNetwork(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4, 4])], {})

