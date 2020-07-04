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
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import numpy as np


from torch.utils.data import DataLoader


from torch import nn


import scipy


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_noahgolmant_pytorch_hessian_eigenthings(_paritybench_base):
    pass
