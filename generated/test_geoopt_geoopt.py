import sys
_module = sys.modules[__name__]
del sys
conf = _module
distance = _module
distance2plane = _module
gyrovector_parallel_transport = _module
mobius_add = _module
mobius_matvec = _module
mobius_sigmoid_apply = _module
parallel_transport = _module
mobius_linear_example = _module
geoopt = _module
docutils = _module
linalg = _module
_expm = _module
batch_linalg = _module
manifolds = _module
base = _module
birkhoff_polytope = _module
euclidean = _module
product = _module
scaled = _module
sphere = _module
stereographic = _module
manifold = _module
math = _module
stiefel = _module
optim = _module
mixin = _module
radam = _module
rsgd = _module
sparse_radam = _module
sparse_rsgd = _module
samplers = _module
rhmc = _module
rsgld = _module
sgrhmc = _module
tensor = _module
utils = _module
setup = _module
test_adam = _module
test_birkhoff = _module
test_euclidean = _module
test_gyrovector_math = _module
test_manifold_basic = _module
test_origin = _module
test_product_manifold = _module
test_random = _module
test_rhmc = _module
test_rsgd = _module
test_scaling = _module
test_sparse_adam = _module
test_sparse_rsgd = _module
test_tensor_api = _module
test_utils = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch.nn


import abc


import itertools


from typing import Optional


from typing import Tuple


from typing import Union


import functools


import inspect


import torch


import types


from typing import List


import numpy as np


def create_ball(ball=None, c=None):
    """
    Helper to create a PoincareBall.

    Sometimes you may want to share a manifold across layers, e.g. you are using scaled PoincareBall.
    In this case you will require same curvature parameters for different layers or end up with nans.

    Parameters
    ----------
    ball : geoopt.PoincareBall
    c : float

    Returns
    -------
    geoopt.PoincareBall
    """
    if ball is None:
        assert c is not None, 'curvature of the ball should be explicitly specified'
        ball = geoopt.PoincareBall(c)
    return ball


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_geoopt_geoopt(_paritybench_base):
    pass
