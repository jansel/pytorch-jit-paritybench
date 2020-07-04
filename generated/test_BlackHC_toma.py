import sys
_module = sys.modules[__name__]
del sys
setup = _module
test_benchmark = _module
test_cpu_mem_limit = _module
test_explicit_toma = _module
test_simple_toma = _module
test_stacktrace = _module
test_toma = _module
toma = _module
batchsize_cache = _module
cpu_memory = _module
stacktrace = _module
torch_cuda_memory = _module

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

class Test_BlackHC_toma(_paritybench_base):
    pass
