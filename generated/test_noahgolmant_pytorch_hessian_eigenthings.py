import sys
_module = sys.modules[__name__]
del sys
main = _module
hessian_eigenthings = _module
hvp_operator = _module
lanczos = _module
power_iter = _module
spectral_density = _module
utils = _module
setup = _module
principle_eigenvec_tests = _module
random_matrix_tests = _module
variance_tests = _module

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


import numpy as np


from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator


from scipy.sparse.linalg import eigsh


from warnings import warn


from torch.utils.data import DataLoader


from torch import nn


import matplotlib.pyplot as plt


import scipy

