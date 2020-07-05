import sys
_module = sys.modules[__name__]
del sys
dsntnn = _module
setup = _module
tests = _module
conftest = _module
test_dsnt = _module
test_euclidean_loss = _module
test_flat_softmax = _module
test_make_gauss = _module
test_regularization = _module

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


from functools import reduce


import torch


import torch.nn.functional


from torch import nn


from torch import onnx


from torch.nn.functional import mse_loss


from torch.testing import assert_allclose

