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


import numpy as np


from collections import OrderedDict


import numpy


import inspect

