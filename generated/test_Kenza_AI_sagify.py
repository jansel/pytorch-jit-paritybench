import sys
_module = sys.modules[__name__]
del sys
sagify = _module
api = _module
build = _module
cloud = _module
hyperparameter_tuning = _module
initialize = _module
local = _module
push = _module
commands = _module
configure = _module
custom_validators = _module
validators = _module
config = _module
log = _module
sagemaker = _module
sagify_base = _module
prediction = _module
predict = _module
predictor = _module
wsgi = _module
training = _module
setup = _module
tests = _module
test_validators = _module
test_build = _module
test_cloud = _module
test_configure = _module
test_initialize = _module
test_local = _module
test_push = _module
test_config = _module
test_sagemaker = _module

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

