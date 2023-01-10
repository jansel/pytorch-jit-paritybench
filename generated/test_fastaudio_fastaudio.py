import sys
_module = sys.modules[__name__]
del sys
setup = _module
fastaudio = _module
all = _module
augment = _module
functional = _module
preprocess = _module
signal = _module
spectrogram = _module
ci = _module
core = _module
config = _module
signal = _module
spectrogram = _module
util = _module
conftest = _module
test_augment = _module
test_config = _module
test_core = _module
test_functional = _module
test_import_lib = _module
test_kwargs = _module
test_run_notebooks = _module
test_signal_augment = _module
test_spectrogram_augment = _module
test_util = _module
test_warnings = _module

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


import torch.fft


from torch import Tensor


from enum import Enum


from scipy.signal import resample_poly


import warnings


from torch.nn import functional as F


from torchaudio.functional import compute_deltas


import random


import torchaudio


from collections import OrderedDict


from inspect import signature


from torch import nn


from functools import wraps


from math import pi


import inspect


import math


from torchaudio.transforms import MelSpectrogram

