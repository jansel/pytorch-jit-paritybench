import sys
_module = sys.modules[__name__]
del sys
asdfghjkl = _module
core = _module
fisher = _module
fr = _module
gradient = _module
hessian = _module
kernel = _module
matrices = _module
mvp = _module
operations = _module
batchnorm = _module
bias = _module
conv = _module
linear = _module
operation = _module
scale = _module
precondition = _module
symmatrix = _module
utils = _module
setup = _module
test_conjugate = _module
test_ifvp = _module
test_mvp = _module

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


from typing import List


import torch.nn as nn


import numpy as np


import torch


import torch.nn.functional as F


from torch.utils.data.dataloader import DataLoader


from torch.utils.data import Subset


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.nn.utils import parameters_to_vector


from torch.nn.utils import vector_to_parameters


from functools import partial


from torch import Tensor


from torch.nn import functional as F


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


import copy


import math


import warnings


from torch import nn


from torch.cuda import nvtx


import time


class Bias(nn.Module):

    def __init__(self):
        super(Bias, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def reset_parameters(self):
        nn.init.constant_(self.weight, 0)

    def forward(self, input):
        return input + self.weight


class Scale(nn.Module):

    def __init__(self):
        super(Scale, self).__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def reset_parameters(self):
        nn.init.constant_(self.weight, 1)

    def forward(self, input):
        return self.weight * input


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Bias,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Scale,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_kazukiosawa_asdl(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

