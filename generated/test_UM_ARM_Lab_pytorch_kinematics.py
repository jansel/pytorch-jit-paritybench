import sys
_module = sys.modules[__name__]
del sys
pytorch_kinematics = _module
cfg = _module
chain = _module
frame = _module
jacobian = _module
mjcf = _module
sdf = _module
transforms = _module
math = _module
rotation_conversions = _module
so3 = _module
transform3d = _module
urdf = _module
urdf_parser_py = _module
xml_reflection = _module
basics = _module
core = _module
setup = _module
tests = _module
test_jacobian = _module
test_kinematics = _module
test_rotation_conversions = _module
test_transform = _module

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


import math


from typing import Tuple


from typing import Union


import functools


from typing import Optional


import torch.nn.functional as F


import warnings


import typing

