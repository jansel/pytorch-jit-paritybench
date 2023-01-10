import sys
_module = sys.modules[__name__]
del sys
alaas = _module
client = _module
utils = _module
metric = _module
server = _module
executors = _module
al_torch = _module
cl_torch = _module
preprocessor = _module
image = _module
text = _module
strategy = _module
base = _module
cluster = _module
diversity = _module
random = _module
uncertainty = _module
util = _module
cfg_manager = _module
data_util = _module
types = _module
models = _module
al_model = _module
al_strategy = _module
config = _module
version = _module
examples = _module
image_clf_client = _module
image_clf_server = _module
text_clf_client = _module
text_clf_server = _module
setup = _module
tests = _module

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


import torch.jit


import torch.nn.functional as F


import warnings


import numpy as np

