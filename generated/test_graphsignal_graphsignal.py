import sys
_module = sys.modules[__name__]
del sys
graphsignal = _module
agent = _module
agent_test = _module
callbacks = _module
keras = _module
keras_test = _module
pytorch_lightning = _module
pytorch_lightning_test = _module
data = _module
builtin_types = _module
builtin_types_test = _module
data_profiler = _module
missing_value_detector = _module
missing_value_detector_test = _module
numpy_ndarray = _module
numpy_ndarray_test = _module
tf_tensor = _module
tf_tensor_test = _module
torch_tensor = _module
torch_tensor_test = _module
endpoint_trace = _module
endpoint_trace_test = _module
graphsignal_async_test = _module
graphsignal_test = _module
proto = _module
signals_pb2 = _module
proto_utils = _module
proto_utils_test = _module
recorders = _module
base_recorder = _module
cprofile_recorder = _module
cprofile_recorder_test = _module
jax_recorder = _module
jax_recorder_test = _module
nvml_recorder = _module
nvml_recorder_test = _module
onnxruntime_recorder = _module
onnxruntime_recorder_test = _module
process_recorder = _module
process_recorder_test = _module
pytorch_recorder = _module
pytorch_recorder_test = _module
tensorflow_recorder = _module
tensorflow_recorder_test = _module
xgboost_recorder = _module
xgboost_recorder_test = _module
trace_sampler = _module
trace_sampler_test = _module
uploader = _module
uploader_test = _module
vendor = _module
pynvml = _module
version = _module
setup = _module

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


import logging


import random


import time


import uuid


from typing import Optional


from torch import Tensor


import functools


import numpy as np


import torch


import torch.distributed as dist

