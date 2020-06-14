import sys
_module = sys.modules[__name__]
del sys
main = _module
opt = _module
src = _module
camera = _module
data_process = _module
datasets = _module
human36m = _module
log = _module
misc = _module
model = _module
procrustes = _module
utils = _module

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


import time


import numpy as np


import torch


import torch.nn as nn


import torch.optim


import torch.backends.cudnn as cudnn


from torch.utils.data import DataLoader


from torch.autograd import Variable


class Linear(nn.Module):

    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)
        out = x + y
        return out


class LinearModel(nn.Module):

    def __init__(self, linear_size=1024, num_stage=2, p_dropout=0.5):
        super(LinearModel, self).__init__()
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.input_size = 16 * 2
        self.output_size = 16 * 3
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)
        self.w2 = nn.Linear(self.linear_size, self.output_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        y = self.w2(y)
        return y


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_weigq_3d_pose_baseline_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(Linear(*[], **{'linear_size': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(LinearModel(*[], **{}), [torch.rand([32, 32])], {})

