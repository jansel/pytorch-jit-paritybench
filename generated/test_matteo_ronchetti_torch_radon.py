import sys
_module = sys.modules[__name__]
del sys
auto_install = _module
benchmark = _module
plot_results = _module
bench = _module
bench = _module
build = _module
build_tools = _module
generate_source = _module
conf = _module
end_to_end = _module
fbp = _module
invisible = _module
landweber = _module
utils = _module
visual_sample = _module
setup = _module
tests = _module
astra_wrapper = _module
sample = _module
test_fanbeam = _module
test_noise = _module
test_parallel_beam = _module
test_shearlet = _module
test_torch = _module
torch_radon = _module
differentiable_functions = _module
filtering = _module
shearlet = _module
solvers = _module
utils = _module
create_build_script = _module

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


import numpy as np


import torch


import matplotlib.pyplot as plt


import torch.nn as nn


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.autograd import gradcheck


import scipy.stats


import abc


import torch.nn.functional as F


import warnings


from torch.autograd import Function

