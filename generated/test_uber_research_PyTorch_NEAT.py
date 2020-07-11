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


import numpy as np


import torch


import torch.nn.functional as F

