import sys
_module = sys.modules[__name__]
del sys
hyperdash = _module
client = _module
code_runner = _module
commands = _module
constants = _module
experiment = _module
hyper_dash = _module
io_buffer = _module
jupyter = _module
jupyter_2_exec = _module
jupyter_3_exec = _module
monitor = _module
sdk_message = _module
server_manager = _module
utils = _module
hyperdash_cli = _module
cli = _module
setup = _module
mocks = _module
test_buffer = _module
test_cli = _module
test_jupyter = _module
test_script_for_run_test = _module
test_sdk = _module

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

