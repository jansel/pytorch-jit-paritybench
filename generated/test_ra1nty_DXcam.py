import sys
_module = sys.modules[__name__]
del sys
d3dshot_max_fps = _module
dxcam_capture = _module
dxcam_max_fps = _module
mss_max_fps = _module
dxcam = _module
_libs = _module
d3d11 = _module
dxgi = _module
user32 = _module
core = _module
device = _module
duplicator = _module
output = _module
stagesurf = _module
processor = _module
base = _module
numpy_processor = _module
util = _module
io = _module
timer = _module
capture_to_video = _module
instant_replay = _module

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

