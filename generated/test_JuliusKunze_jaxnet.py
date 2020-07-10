import sys
_module = sys.modules[__name__]
del sys
examples = _module
mnist_classifier = _module
mnist_vae = _module
ocr_rnn = _module
pixelcnn = _module
policy_gradient = _module
resnet50 = _module
wavenet = _module
jaxnet = _module
core = _module
modules = _module
optimizers = _module
setup = _module
tests = _module
test_core = _module
test_examples = _module
test_modules = _module
test_optimizers = _module
util = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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

