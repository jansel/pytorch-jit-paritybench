import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
test_automata = _module
test_cross = _module
test_derivatives = _module
test_gpu = _module
test_indexing = _module
test_init = _module
test_ops = _module
test_round = _module
test_tensor = _module
test_tools = _module
util = _module
tntorch = _module
anova = _module
autodiff = _module
automata = _module
create = _module
cross = _module
derivatives = _module
logic = _module
metrics = _module
ops = _module
round = _module
tensor = _module
tools = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'

