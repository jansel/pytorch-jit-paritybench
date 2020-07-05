import sys
_module = sys.modules[__name__]
del sys
master = _module
args = _module
config_reader = _module
spert = _module
entities = _module
evaluator = _module
input_reader = _module
loss = _module
models = _module
opt = _module
sampling = _module
spert_trainer = _module
trainer = _module
util = _module

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


from abc import ABC


import torch


import math


from torch.nn import DataParallel


from torch.optim import Optimizer


from torch.utils.data import DataLoader


import logging


from typing import List


from typing import Dict


from typing import Tuple

