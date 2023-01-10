import sys
_module = sys.modules[__name__]
del sys
date_extraction = _module
evaluate_famous_models = _module
evaluate_rnn_models = _module
setup = _module
tests = _module
test_conv2d = _module
test_matmul = _module
test_relu = _module
test_utils = _module
thop = _module
__version__ = _module
fx_profile = _module
onnx_profile = _module
profile = _module
rnn_hooks = _module
utils = _module
vision = _module
basic_hooks = _module
calc_func = _module
efficientnet = _module
onnx_counter = _module

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


import torch


from torchvision import models


import torch.nn as nn


import logging


import torch as th


from torch.fx import symbolic_trace


from torch.fx.passes.shape_prop import ShapeProp


import torch.nn


import numpy as np


from torch.nn.utils.rnn import PackedSequence


from torch.nn.modules.conv import _ConvNd


import warnings

