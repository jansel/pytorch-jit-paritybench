import sys
_module = sys.modules[__name__]
del sys
pytorch_memlab = _module
courtesy = _module
line_profiler = _module
extension = _module
line_records = _module
profile = _module
mem_reporter = _module
utils = _module
setup = _module
test = _module
test_courtesy = _module
test_line_profiler = _module
test_mem_reporter = _module

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


import torch


from functools import wraps


import math


from collections import defaultdict


import numpy as np


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Stonesjtu_pytorch_memlab(_paritybench_base):
    pass
