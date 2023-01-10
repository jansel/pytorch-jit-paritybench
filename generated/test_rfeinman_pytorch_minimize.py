import sys
_module = sys.modules[__name__]
del sys
conf = _module
scipy_benchmark = _module
train_mnist_Minimizer = _module
setup = _module
tests = _module
test_imports = _module
torchmin = _module
test_leastsquares = _module
benchmarks = _module
bfgs = _module
cg = _module
function = _module
line_search = _module
lstsq = _module
cg = _module
common = _module
least_squares = _module
linear_operator = _module
lsmr = _module
trf = _module
minimize = _module
minimize_constr = _module
newton = _module
optim = _module
minimizer = _module
scipy_minimizer = _module
trustregion = _module
base = _module
dogleg = _module
exact = _module
krylov = _module
ncg = _module

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


from scipy import optimize


import matplotlib.pyplot as plt


import torch.nn as nn


import torch.nn.functional as F


from torchvision import datasets


from abc import ABC


from abc import abstractmethod


from torch import Tensor


from scipy.optimize import OptimizeResult


from typing import List


from typing import Optional


from collections import namedtuple


import torch.autograd as autograd


from torch._vmap_internals import _vmap


import warnings


from torch.optim.lbfgs import _strong_wolfe


from torch.optim.lbfgs import _cubic_interpolate


from scipy.optimize import minimize_scalar


import numpy as np


from scipy.sparse.linalg import LinearOperator


from warnings import warn


import numbers


from scipy.optimize._lsq.common import print_header_nonlinear


from scipy.optimize._lsq.common import print_iteration_nonlinear


from scipy.optimize import minimize


from scipy.optimize import Bounds


from scipy.optimize import NonlinearConstraint


from scipy.sparse.linalg import eigsh


from functools import reduce


from torch.optim import Optimizer


from torch.autograd.functional import _construct_standard_basis_for


from torch.autograd.functional import _grad_postprocess


from torch.autograd.functional import _tuple_postprocess


from torch.autograd.functional import _as_tuple


from torch.linalg import norm


from typing import Tuple


from scipy.linalg import get_lapack_funcs


from scipy.linalg import eigh_tridiagonal

