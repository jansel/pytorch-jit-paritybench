import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
_linear_system_fixtures = _module
_linear_system_models = _module
test_filters = _module
test_jacobians = _module
test_split_trajectories = _module
test_train_component = _module
test_train_e2e = _module
test_unscented_transform = _module
torchfilter = _module
base = _module
_dynamics_model = _module
_filter = _module
_kalman_filter_base = _module
_kalman_filter_measurement_model = _module
_particle_filter_measurement_model = _module
_virtual_sensor_model = _module
data = _module
_particle_filter_measurement_dataset = _module
_single_step_dataset = _module
_split_trajectories = _module
_subsequence_dataset = _module
filters = _module
_extended_information_filter = _module
_extended_kalman_filter = _module
_particle_filter = _module
_square_root_unscented_kalman_filter = _module
_unscented_kalman_filter = _module
_virtual_sensor_filters = _module
train = _module
_train_dynamics = _module
_train_filter = _module
_train_kalman_filter_measurement = _module
_train_particle_filter_measurement = _module
_train_virtual_sensor = _module
types = _module
utils = _module
_sigma_points = _module
_unscented_transform = _module

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


from typing import List


from typing import Dict


from typing import Tuple


import numpy as np


import torch


from typing import cast


import torch.nn as nn


import abc


import scipy.stats


from torch.utils.data import Dataset


from typing import Optional


import torch.nn.functional as F


from torch.utils.data import DataLoader


from typing import Callable


from typing import NamedTuple


from typing import Union


import warnings

