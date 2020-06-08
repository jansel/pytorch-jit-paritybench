import sys
_module = sys.modules[__name__]
del sys
main = _module
models = _module
plot = _module
sem = _module
penalties = _module

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


import numpy as np


import torch


from torch import nn


from torch import optim


from torch import autograd


import math


from itertools import chain


from itertools import combinations


from scipy.stats import f as fdist


from scipy.stats import ttest_ind


from torch.autograd import grad


import scipy.optimize


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_facebookresearch_InvariantRiskMinimization(_paritybench_base):
    pass
