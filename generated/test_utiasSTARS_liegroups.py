import sys
_module = sys.modules[__name__]
del sys
conf = _module
liegroups = _module
_base = _module
numpy = _module
se2 = _module
se3 = _module
so2 = _module
so3 = _module
_base = _module
se2 = _module
se3 = _module
so2 = _module
so3 = _module
utils = _module
setup = _module
test_se2_numpy = _module
test_se3_numpy = _module
test_so2_numpy = _module
test_so3_numpy = _module
test_se2_torch = _module
test_se3_torch = _module
test_so2_torch = _module
test_so3_torch = _module
test_utils_torch = _module

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


import torch


import numpy as np


import copy

