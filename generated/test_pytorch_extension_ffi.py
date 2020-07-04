import sys
_module = sys.modules[__name__]
del sys
build = _module
my_package = _module
functions = _module
add = _module
modules = _module
add = _module
setup = _module
test = _module
add = _module
test = _module

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


from torch.nn.modules.module import Module


import torch


import torch.nn as nn


from torch.autograd import Variable


class MyAddFunction(Function):

    def forward(self, input1, input2):
        output = input1.new()
        if not input1.is_cuda:
            my_lib.my_lib_add_forward(input1, input2, output)
        else:
            my_lib.my_lib_add_forward_cuda(input1, input2, output)
        return output

    def backward(self, grad_output):
        grad_input = grad_output.new()
        if not grad_output.is_cuda:
            my_lib.my_lib_add_backward(grad_output, grad_input)
        else:
            my_lib.my_lib_add_backward_cuda(grad_output, grad_input)
        return grad_input


class MyAddModule(Module):

    def forward(self, input1, input2):
        return MyAddFunction()(input1, input2)


class MyNetwork(nn.Module):

    def __init__(self):
        super(MyNetwork, self).__init__()
        self.add = MyAddModule()

    def forward(self, input1, input2):
        return self.add(input1, input2)


class MyAddModule(Module):

    def forward(self, input1, input2):
        return MyAddFunction()(input1, input2)


class MyNetwork(nn.Module):

    def __init__(self):
        super(MyNetwork, self).__init__()
        self.add = MyAddModule()

    def forward(self, input1, input2):
        return self.add(input1, input2)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_pytorch_extension_ffi(_paritybench_base):
    pass
