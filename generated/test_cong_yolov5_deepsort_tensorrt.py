import sys
_module = sys.modules[__name__]
del sys
deep_sort = _module
deep = _module
feature_extractor_trt = _module
deep_sort_trt = _module
sort = _module
detection = _module
iou_matching = _module
kalman_filter = _module
linear_assignment = _module
nn_matching = _module
preprocessing = _module
track = _module
tracker = _module
utils = _module
asserts = _module
draw = _module
evaluation = _module
io = _module
json_logger = _module
log = _module
parser = _module
tools = _module
demo_trt = _module
detector_trt = _module
tracker_trt = _module

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


import numpy as np

