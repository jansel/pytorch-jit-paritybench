import sys
_module = sys.modules[__name__]
del sys
profile_torchvision = _module
profile_transformer = _module
trace_linear = _module
setup = _module
torchprofile = _module
handlers = _module
profile = _module
utils = _module
flatten = _module
ir = _module
graph = _module
node = _module
variable = _module
math = _module
trace = _module
version = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


from torchvision import models


from torch.nn.modules.transformer import Transformer


import torch.nn as nn


from collections import deque


import warnings


import torch.jit


def flatten(inputs):
    queue = deque([inputs])
    outputs = []
    while queue:
        x = queue.popleft()
        if isinstance(x, (list, tuple)):
            queue.extend(x)
        elif isinstance(x, dict):
            queue.extend(x.values())
        elif isinstance(x, torch.Tensor):
            outputs.append(x)
    return outputs


class Flatten(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return flatten(outputs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Flatten,
     lambda: ([], {'model': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
]

class Test_mit_han_lab_torchprofile(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

