import sys
_module = sys.modules[__name__]
del sys
neptune = _module
livelossplot = _module
inputs = _module
generic_keras = _module
keras = _module
poutyne = _module
pytorch_ignite = _module
tf_keras = _module
main_logger = _module
outputs = _module
base_output = _module
bokeh_plot = _module
extrema_printer = _module
matplotlib_plot = _module
matplotlib_subplots = _module
neptune_logger = _module
tensorboard_logger = _module
tensorboard_tf_logger = _module
plot_losses = _module
version = _module
setup = _module
external_api_test_neptune = _module
external_test_examples = _module
external_test_keras = _module
external_test_pytorch_ignite = _module
external_test_tensorboard = _module
test_bokeh_output = _module
test_extrema_printer = _module
test_main_logger = _module
test_plot_losses = _module

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

