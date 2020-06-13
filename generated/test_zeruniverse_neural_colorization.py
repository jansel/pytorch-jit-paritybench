import sys
_module = sys.modules[__name__]
del sys
build_dataset_directory = _module
colorize = _module
model = _module
resize_all_imgs = _module
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


import torch


import torch.nn as nn


from functools import reduce


from torch.autograd import Variable


from torch.utils import data


import numpy as np


class shave_block(nn.Module):

    def __init__(self, s):
        super(shave_block, self).__init__()
        self.s = s

    def forward(self, x):
        return x[:, :, self.s:-self.s, self.s:-self.s]


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

class Test_zeruniverse_neural_colorization(_paritybench_base):
    pass
    def test_000(self):
        self._check(LambdaBase(*[], **{'fn': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(shave_block(*[], **{'s': 4}), [torch.rand([4, 4, 4, 4])], {})

