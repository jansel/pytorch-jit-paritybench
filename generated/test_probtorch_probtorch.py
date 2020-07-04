import sys
_module = sys.modules[__name__]
del sys
conf = _module
probtorch = _module
objectives = _module
importance = _module
marginal = _module
montecarlo = _module
stochastic = _module
util = _module
version = _module
setup = _module
test = _module
common = _module
test_stochastic = _module
test_util = _module

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


from numbers import Number


from torch.nn.functional import softmax


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_probtorch_probtorch(_paritybench_base):
    pass
