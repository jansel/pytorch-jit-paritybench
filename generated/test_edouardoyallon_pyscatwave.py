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

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import math


import torch


import torch.optim


from torchvision.datasets.mnist import MNIST


from torch.autograd import Variable


import torch.nn.functional as F


import numpy as np


import torchvision.transforms as transforms


from torch.autograd import Function


from collections import defaultdict


from collections import namedtuple


import scipy.fftpack as fft


import warnings


from torch.nn import ReflectionPad2d as pad_function


from string import Template

