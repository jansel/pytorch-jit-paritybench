import sys
_module = sys.modules[__name__]
del sys
mnist = _module
test = _module
hessian = _module
gradient = _module
hessian = _module
power_method = _module
rayleigh_quotient = _module
setup = _module

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


import torch.nn as nn


import torch.nn.functional as F


import torch.utils.data


import torchvision


from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(5, 5, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(5, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), x.size(1), -1).mean(-1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
]

class Test_mariogeiger_hessian(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

