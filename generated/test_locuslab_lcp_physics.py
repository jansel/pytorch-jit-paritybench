import sys
_module = sys.modules[__name__]
del sys
demo = _module
fixed_joint_demo = _module
grad_demo = _module
encoder = _module
learn_params = _module
run_breakout = _module
inference = _module
lcp_physics = _module
lcp = _module
lcp = _module
solvers = _module
dev_pdipm = _module
pdipm = _module
util = _module
physics = _module
bodies = _module
constraints = _module
contacts = _module
engines = _module
forces = _module
utils = _module
world = _module
setup = _module
test_bodies = _module
test_demos = _module
test_hull = _module

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


import math


import torch


from torch.autograd import Variable


import time


import numpy as np


from torch.nn import MSELoss


from torch.optim import RMSprop


from torch import nn


from torch.autograd import Function


from enum import Enum


from scipy.linalg import lu_factor


from scipy.linalg import lu_solve


from scipy.sparse import csc_matrix


from scipy.sparse import eye


from scipy.sparse import hstack


from scipy.sparse import vstack


from scipy.sparse import diags


from scipy.sparse import block_diag


from scipy.sparse import bmat


from scipy.sparse.linalg import splu


from scipy.sparse.linalg import spsolve


import random


from functools import lru_cache

