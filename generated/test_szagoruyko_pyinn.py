import sys
_module = sys.modules[__name__]
del sys
pyinn = _module
cdgmm = _module
conv2d_depthwise = _module
dgmm = _module
im2col = _module
modules = _module
ncrelu = _module
utils = _module
setup = _module
benchmark = _module
test = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


from torch.autograd import Function


import torch


from torch.nn.modules.utils import _pair


import torch.nn.functional as F


from torch.autograd import Variable


from torch.nn.init import kaiming_normal


from torch.backends import cudnn


from functools import partial


from torch.autograd import gradcheck

