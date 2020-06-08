import sys
_module = sys.modules[__name__]
del sys
mnist = _module
reconstruction_exp = _module
FFT = _module
scatwave = _module
differentiable = _module
filters_bank = _module
scattering = _module
utils = _module
setup = _module
test_scattering = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import math


import torch


import torch.optim


from torch.autograd import Variable


import torch.nn.functional as F


import warnings


from torch.nn import ReflectionPad2d as pad_function


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_edouardoyallon_pyscatwave(_paritybench_base):
    pass
