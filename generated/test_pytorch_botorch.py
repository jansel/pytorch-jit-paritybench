import sys
_module = sys.modules[__name__]
del sys
botorch = _module
acquisition = _module
acquisition = _module
active_learning = _module
analytic = _module
cost_aware = _module
fixed_feature = _module
knowledge_gradient = _module
max_value_entropy_search = _module
monte_carlo = _module
objective = _module
utils = _module
cross_validation = _module
exceptions = _module
errors = _module
warnings = _module
fit = _module
gen = _module
generation = _module
gen = _module
sampling = _module
logging = _module
models = _module
converter = _module
cost = _module
deterministic = _module
gp_regression = _module
gp_regression_fidelity = _module
gpytorch = _module
kernels = _module
downsampling = _module
exponential_decay = _module
linear_truncated_fidelity = _module
model = _module
model_list_gp_regression = _module
multitask = _module
pairwise_gp = _module
transforms = _module
input = _module
outcome = _module
optim = _module
fit = _module
initializers = _module
numpy_converter = _module
optimize = _module
parameter_constraints = _module
stopping = _module
posteriors = _module
posterior = _module
transformed = _module
pairwise_samplers = _module
qmc = _module
samplers = _module
settings = _module
test_functions = _module
base = _module
multi_fidelity = _module
synthetic = _module
constraints = _module
feasible_volume = _module
testing = _module
parse_sphinx = _module
parse_tutorials = _module
patch_site_config = _module
run_tutorials = _module
update_versions_html = _module
validate_sphinx = _module
setup = _module
conf = _module
test = _module
test_acquisition = _module
test_active_learning = _module
test_analytic = _module
test_cost_aware = _module
test_fixed_feature = _module
test_knowledge_gradient = _module
test_max_value_entropy_search = _module
test_monte_carlo = _module
test_objective = _module
test_utils = _module
test_errors = _module
test_warnings = _module
test_gen = _module
test_sampling = _module
test_downsampling = _module
test_exponential_decay = _module
test_linear_truncated_fidelity = _module
test_converter = _module
test_cost = _module
test_deterministic = _module
test_gp_regression = _module
test_gp_regression_fidelity = _module
test_gpytorch = _module
test_model = _module
test_model_list_gp_regression = _module
test_multitask = _module
test_pairwise_gp = _module
test_input = _module
test_outcome = _module
test_initializers = _module
test_numpy_converter = _module
test_optimize = _module
test_parameter_constraints = _module
test_stopping = _module
test_posterior = _module
test_transformed = _module
test_pairwise_sampler = _module
test_qmc = _module
test_sampler = _module
test_cross_validation = _module
test_cuda = _module
test_end_to_end = _module
test_fit = _module
test_base = _module
test_multi_fidelity = _module
test_synthetic = _module
test_logging = _module
test_settings = _module
test_constraints = _module
test_feasible_volume = _module
test_testing = _module
test_transforms = _module

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


import warnings


from abc import ABC


from abc import abstractmethod


from typing import Optional


from torch import Tensor


from torch.nn import Module


from typing import Any


from typing import Callable


import torch


from typing import List


from typing import Union


from typing import Dict


from typing import Tuple


from typing import Type


from scipy.optimize import minimize


from torch.optim import Optimizer


from copy import deepcopy


from torch.nn import ModuleDict


from collections import OrderedDict


import time


from typing import NamedTuple


from typing import Set


import numpy as np


from scipy.optimize import Bounds


from torch.optim.adam import Adam


from torch.optim.optimizer import Optimizer


from math import inf


from torch.quasirandom import SobolEngine


import math


class BotorchWarning(Warning):
    """Base botorch warning."""
    pass


class CostAwareUtility(Module, ABC):
    """Abstract base class for cost-aware utilities."""

    @abstractmethod
    def forward(self, X: Tensor, deltas: Tensor, **kwargs: Any) ->Tensor:
        """Evaluate the cost-aware utility on the candidates and improvements.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of with `q` `d`-dim design
                points each for each t-batch.
            deltas: A `num_fantasies x batch_shape`-dim Tensor of `num_fantasy`
                samples from the marginal improvement in utility over the
                current state at `X` for each t-batch.

        Returns:
            A `num_fantasies x batch_shape`-dim Tensor of cost-transformed utilities.
        """
        pass


class AcquisitionObjective(Module, ABC):
    """Abstract base class for objectives."""
    ...


class SamplingStrategy(Module, ABC):
    """Abstract base class for sampling-based generation strategies."""

    @abstractmethod
    def forward(self, X: Tensor, num_samples: int=1, **kwargs: Any) ->Tensor:
        """Sample according to the SamplingStrategy.

        Args:
            X: A `batch_shape x N x d`-dim Tensor from which to sample (in the `N`
                dimension).
            num_samples: The number of samples to draw.
            kwargs: Additional implementation-specific kwargs.

        Returns:
            A `batch_shape x num_samples x d`-dim Tensor of samples from `X`, where
            `X[..., i, :]` is the `i`-th sample.
        """
        pass


class InputTransform(Module, ABC):
    """Abstract base class for input transforms."""

    @abstractmethod
    def forward(self, X: Tensor) ->Tensor:
        """Transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        pass

    def untransform(self, X: Tensor) ->Tensor:
        """Un-transform the inputs to a model.

        Args:
            X: A `batch_shape x n x d`-dim tensor of transformed inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-transformed inputs.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not implement the `untransform` method')


class ChainedInputTransform(InputTransform, ModuleDict):
    """An input transform representing the chaining of individual transforms"""

    def __init__(self, **transforms: InputTransform) ->None:
        """Chaining of input transforms.

        Args:
            transforms: The transforms to chain. Internally, the names of the
                kwargs are used as the keys for accessing the individual
                transforms on the module.
        """
        super().__init__(transforms)

    def forward(self, X: Tensor) ->Tensor:
        """Transform the inputs to a model.

        Individual transforms are applied in sequence.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of transformed inputs.
        """
        for tf in self.values():
            X = tf.forward(X)
        return X

    def untransform(self, X: Tensor) ->Tensor:
        """Un-transform the inputs to a model.

        Un-transforms of the individual transforms are applied in reverse sequence.

        Args:
            X: A `batch_shape x n x d`-dim tensor of transformed inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-transformed inputs.
        """
        for tf in reversed(self.values()):
            X = tf.untransform(X)
        return X


class BaseTestProblem(Module, ABC):
    """Base class for test functions."""
    dim: int
    _bounds: List[Tuple[float, float]]
    _check_grad_at_opt: bool = True

    def __init__(self, noise_std: Optional[float]=None, negate: bool=False) ->None:
        """Base constructor for test functions.

        Arguments:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
        """
        super().__init__()
        self.noise_std = noise_std
        self.negate = negate
        self.register_buffer('bounds', torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2))

    def forward(self, X: Tensor, noise: bool=True) ->Tensor:
        """Evaluate the function on a set of points.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                function.
            noise: If `True`, add observation noise as specified by `noise_std`.

        Returns:
            A `batch_shape`-dim tensor ouf function evaluations.
        """
        batch = X.ndimension() > 1
        X = X if batch else X.unsqueeze(0)
        f = self.evaluate_true(X=X)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        if self.negate:
            f = -f
        return f if batch else f.squeeze(0)

    @abstractmethod
    def evaluate_true(self, X: Tensor) ->Tensor:
        """Evaluate the function (w/o observation noise) on a set of points."""
        pass


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ChainedInputTransform,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_pytorch_botorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

