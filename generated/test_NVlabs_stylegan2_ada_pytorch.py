import sys
_module = sys.modules[__name__]
del sys
calc_metrics = _module
dataset_tool = _module
generate = _module
metrics = _module
projector = _module
style_mixing = _module
custom_ops = _module
bias_act = _module
conv2d_gradfix = _module
conv2d_resample = _module
fma = _module
grid_sample_gradfix = _module
upfirdn2d = _module
training_stats = _module
train = _module
training = _module

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


import copy


import torch


import re


from typing import List


from typing import Optional


import numpy as np


from time import perf_counter


import torch.nn.functional as F


import torch.utils.cpp_extension


from torch.utils.file_baton import FileBaton


import warnings

