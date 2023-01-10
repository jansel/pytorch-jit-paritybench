import sys
_module = sys.modules[__name__]
del sys
conf = _module
ikomia = _module
core = _module
auth = _module
config = _module
task = _module
dataio = _module
dataprocess = _module
datadictIO = _module
displayIO = _module
registry = _module
tile_processing = _module
workflow = _module
dnn = _module
dataset = _module
datasetio = _module
dnntrain = _module
monitoring = _module
datasetmapper = _module
models = _module
utils = _module
autocomplete = _module
data = _module
http = _module
plugintools = _module
pyqtutils = _module
qtconversion = _module
tests = _module
start_train_monitoring = _module
setup = _module

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


from torch.utils.data import Dataset


import numpy as np


from torchvision import models


import logging

