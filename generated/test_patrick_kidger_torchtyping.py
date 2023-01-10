import sys
_module = sys.modules[__name__]
del sys
setup = _module
conftest = _module
test_consistency = _module
test_details = _module
test_dtype_layout = _module
test_ellipsis = _module
test_examples = _module
test_extensions = _module
test_misc = _module
test_shape = _module
torchtyping = _module
pytest_plugin = _module
tensor_details = _module
tensor_type = _module
typechecker = _module
utils = _module

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


import re


import torch


import warnings


from torch import rand


from typing import Union


from torch import ones


from torch import sparse_coo


from torch import tensor


from torch import Tensor


from typing import Tuple


from typing import Any


import abc


import collections


from typing import Optional


from typing import NoReturn


import inspect


from typing import Dict


from typing import List

