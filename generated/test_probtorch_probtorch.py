import sys
_module = sys.modules[__name__]
del sys
conf = _module
probtorch = _module
objectives = _module
importance = _module
marginal = _module
montecarlo = _module
stochastic = _module
util = _module
version = _module
setup = _module
test = _module
common = _module
test_stochastic = _module
test_util = _module

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
xrange = range
wraps = functools.wraps


import torch


from numbers import Number


from torch.nn.functional import softmax


from collections import OrderedDict


from collections import MutableMapping


import abc


from enum import Enum


import re


import math


from functools import wraps


import warnings


from itertools import product


import torch.cuda


from torch.autograd import Variable


from torch.distributions import Normal


from random import randint


from random import sample

