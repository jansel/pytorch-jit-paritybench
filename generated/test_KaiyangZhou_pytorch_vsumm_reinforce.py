import sys
_module = sys.modules[__name__]
del sys
create_split = _module
knapsack = _module
main = _module
models = _module
parse_json = _module
parse_log = _module
rewards = _module
summary2video = _module
utils = _module
visualize_results = _module
vsum_tools = _module

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


import time


import numpy as np


import torch


import torch.nn as nn


import torch.backends.cudnn as cudnn


from torch.optim import lr_scheduler


from torch.distributions import Bernoulli


from torch.nn import functional as F


class DSN(nn.Module):
    """Deep Summarization Network"""

    def __init__(self, in_dim=1024, hid_dim=256, num_layers=1, cell='lstm'):
        super(DSN, self).__init__()
        assert cell in ['lstm', 'gru'], "cell must be either 'lstm' or 'gru'"
        if cell == 'lstm':
            self.rnn = nn.LSTM(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(in_dim, hid_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim * 2, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        p = F.sigmoid(self.fc(h))
        return p


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DSN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 1024])], {}),
     True),
]

class Test_KaiyangZhou_pytorch_vsumm_reinforce(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

