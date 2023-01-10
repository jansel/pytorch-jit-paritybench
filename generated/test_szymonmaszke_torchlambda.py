import sys
_module = sys.modules[__name__]
del sys
performance = _module
setup = _module
gather = _module
model = _module
payload = _module
process = _module
utils = _module
validate = _module
torchlambda = _module
_version = _module
arguments = _module
parser = _module
subparsers = _module
implementation = _module
build = _module
docker = _module
general = _module
layer = _module
template = _module
header = _module
imputation = _module
macro = _module
validator = _module
main = _module
subcommands = _module
settings = _module

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


import random


import torch


import torchvision


import collections


import itertools

