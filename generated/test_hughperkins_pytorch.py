import sys
_module = sys.modules[__name__]
del sys
runner = _module
test_rnn = _module
try_asfloattensor = _module
setup = _module
pybit = _module
PyTorchAug = _module
PyTorchHelpers = _module
PyTorchLua = _module
floattensor = _module
test = _module
testByteTensor = _module
testDoubleTensor = _module
testFloatTensor = _module
testLongTensor = _module
test_call_lua = _module
test_helpers = _module
test_nnx = _module
test_pynn = _module
test_pytorch = _module
test_pytorch_refcount = _module
test_save_load = _module
test_throw = _module
mnist = _module
loader = _module

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
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_hughperkins_pytorch(_paritybench_base):
    pass
