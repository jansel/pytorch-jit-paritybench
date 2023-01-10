import sys
_module = sys.modules[__name__]
del sys
setup = _module
sunstreaker = _module
activations = _module
applications = _module
diffusion = _module
ddpm = _module
transformers = _module
bert = _module
data = _module
engine = _module
base_layer = _module
functional = _module
input_layer = _module
sequential = _module
training = _module
layers = _module
core = _module
normalization = _module
pooling = _module
losses = _module
metrics = _module
models = _module
optimizers = _module
regularizers = _module

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

