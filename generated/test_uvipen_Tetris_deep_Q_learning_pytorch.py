import sys
_module = sys.modules[__name__]
del sys
deep_q_network = _module
tetris = _module
test = _module
train = _module

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


import torch.nn as nn


from random import random


from random import randint


from random import sample


import numpy as np


import torch


from collections import deque


class DeepQNetwork(nn.Module):

    def __init__(self):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_uvipen_Tetris_deep_Q_learning_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(DeepQNetwork(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

