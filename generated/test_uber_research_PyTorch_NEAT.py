import sys
_module = sys.modules[__name__]
del sys
main = _module
pytorch_neat = _module
activations = _module
adaptive_linear_net = _module
adaptive_net = _module
aggregations = _module
cppn = _module
dask_helpers = _module
maze = _module
multi_env_eval = _module
neat_reporter = _module
recurrent_net = _module
strict_t_maze = _module
t_maze = _module
turning_t_maze = _module
tests = _module
test_adaptive_linear = _module
test_cppn = _module
test_maze = _module
test_multi_env_eval = _module
test_recurrent = _module
test_strict_t_maze = _module
test_t_maze = _module
test_turning_t_maze = _module

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


import torch.nn.functional as F


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_uber_research_PyTorch_NEAT(_paritybench_base):
    pass
