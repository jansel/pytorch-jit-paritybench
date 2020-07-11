import sys
_module = sys.modules[__name__]
del sys
date_extraction = _module
evaluate_famous_models = _module
evaluate_rnn_models = _module
setup = _module
thop = _module
profile = _module
rnn_hooks = _module
utils = _module
vision = _module
basic_hooks = _module
efficientnet = _module

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


from torch.nn.modules.conv import _ConvNd

