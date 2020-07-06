import sys
_module = sys.modules[__name__]
del sys
EmptyOP = _module
benchmark_empty_op = _module
benchmark_roi_align = _module
MulElemWise = _module
test_mul_func = _module
test_mul_op = _module
ConstantOP = _module
MyFirstOP = _module
RunROIAlign = _module
TVMOp = _module
AdditionOP = _module
dynamic_import_op = _module
mobula = _module
build = _module
build_utils = _module
config = _module
const = _module
dtype = _module
edict = _module
func = _module
glue = _module
backend = _module
common = _module
cp = _module
mx = _module
np = _module
th = _module
op = _module
custom = _module
gen_code = _module
loader = _module
register = _module
testing = _module
utils = _module
version = _module
ROIAlign = _module
Softmax = _module
test_softmax = _module
setup = _module
test_gpu = _module
test_config = _module
test_ctx = _module
test_examples = _module
test_func = _module
test_constant_op = _module
test_addition = _module
test_template = _module
test_roi_align_op = _module
test_testing = _module
test_utils = _module
autoformat = _module
clean_all_build = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from itertools import chain


import torch

