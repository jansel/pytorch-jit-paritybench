import sys
_module = sys.modules[__name__]
del sys
convert_torch = _module

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


import torch.optim as optim


from torch.autograd import Variable


import numpy as np


import math


from functools import reduce


class LambdaBase(nn.Sequential):

    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_fanq15_caffe_to_torch_to_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(LambdaBase(*[], **{'fn': 4}), [torch.rand([4, 4, 4, 4])], {})

