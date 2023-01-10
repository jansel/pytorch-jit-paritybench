import sys
_module = sys.modules[__name__]
del sys
gather = _module
scatter_segment = _module
conf = _module
setup = _module
test_logsumexp = _module
test_softmax = _module
test_std = _module
test_broadcasting = _module
test_gather = _module
test_multi_gpu = _module
test_scatter = _module
test_segment = _module
test_zero_tensors = _module
torch_scatter = _module
composite = _module
logsumexp = _module
softmax = _module
std = _module
placeholder = _module
scatter = _module
segment_coo = _module
segment_csr = _module
testing = _module
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


import time


import itertools


import torch


from scipy.io import loadmat


from itertools import product


from torch.__config__ import parallel_info


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.autograd import gradcheck


from typing import Optional


from typing import Tuple


from typing import Any

