import sys
_module = sys.modules[__name__]
del sys
SumTree = _module
cartpole_per = _module
prioritized_memory = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import random


import numpy as np


from collections import deque


import torch.nn as nn


import torch.optim as optim


import torch.nn.functional as F


from torch.autograd import Variable


from torchvision import transforms


class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(nn.Linear(state_size, 24), nn.ReLU(), nn.
            Linear(24, 24), nn.ReLU(), nn.Linear(24, action_size))

    def forward(self, x):
        return self.fc(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_rlcode_per(_paritybench_base):
    pass
    def test_000(self):
        self._check(DQN(*[], **{'state_size': 4, 'action_size': 4}), [torch.rand([4, 4, 4, 4])], {})

