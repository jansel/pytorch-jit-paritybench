import sys
_module = sys.modules[__name__]
del sys
example = _module
mlogger = _module
config = _module
container = _module
defaults = _module
metric = _module
base = _module
history = _module
to_float = _module
plotter = _module
graph = _module
text = _module
visdom_plotter = _module
stdout = _module
setup = _module
test = _module
test_config = _module
test_container = _module
test_metrics = _module

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

class Test_oval_group_mlogger(_paritybench_base):
    pass
