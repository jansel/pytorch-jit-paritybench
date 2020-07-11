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
utils = _module
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
utils = _module
utils = _module
optim = _module
fit = _module
initializers = _module
numpy_converter = _module
optimize = _module
parameter_constraints = _module
stopping = _module
utils = _module
posteriors = _module
deterministic = _module
gpytorch = _module
posterior = _module
transformed = _module
sampling = _module
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
objective = _module
sampling = _module
testing = _module
transforms = _module
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
test_utils = _module
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
test_utils = _module
test_input = _module
test_outcome = _module
test_utils = _module
test_initializers = _module
test_numpy_converter = _module
test_optimize = _module
test_parameter_constraints = _module
test_stopping = _module
test_utils = _module
test_deterministic = _module
test_gpytorch = _module
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
test_objective = _module
test_sampling = _module
test_testing = _module
test_transforms = _module

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


import warnings


from abc import ABC


from abc import abstractmethod


from typing import Optional


from torch import Tensor


from torch.nn import Module


from copy import deepcopy


from typing import Dict


from typing import Tuple


from typing import Union


import torch


from torch.distributions import Normal


from typing import Any


from typing import Callable


from typing import List


from math import log


from scipy.optimize import brentq


import math


from torch.quasirandom import SobolEngine


from typing import NamedTuple


from typing import Type


from scipy.optimize import minimize


from torch.optim import Optimizer


import typing


import itertools


from typing import Iterator


import numpy as np


from scipy import optimize


from torch import float32


from torch import float64


from torch.nn import ModuleDict


from collections import OrderedDict


import time


from typing import Set


from scipy.optimize import Bounds


from torch.optim.adam import Adam


from torch.optim.optimizer import Optimizer


from math import inf


from functools import partial


from inspect import signature


from abc import abstractproperty


from itertools import combinations


from typing import Generator


from torch import LongTensor


from functools import wraps


import re


from random import random


from scipy.stats import shapiro


from itertools import chain


class BotorchWarning(Warning):
    """Base botorch warning."""
    pass


class Posterior(ABC):
    """Abstract base class for botorch posteriors."""

    @abstractproperty
    def device(self) ->torch.device:
        """The torch device of the posterior."""
        pass

    @abstractproperty
    def dtype(self) ->torch.dtype:
        """The torch dtype of the posterior."""
        pass

    @abstractproperty
    def event_shape(self) ->torch.Size:
        """The event shape (i.e. the shape of a single sample)."""
        pass

    @property
    def mean(self) ->Tensor:
        """The mean of the posterior as a `(b) x n x m`-dim Tensor."""
        raise NotImplementedError(f'Property `mean` not implemented for {self.__class__.__name__}')

    @property
    def variance(self) ->Tensor:
        """The variance of the posterior as a `(b) x n x m`-dim Tensor."""
        raise NotImplementedError(f'Property `variance` not implemented for {self.__class__.__name__}')

    @abstractmethod
    def rsample(self, sample_shape: Optional[torch.Size]=None, base_samples: Optional[Tensor]=None) ->Tensor:
        """Sample from the posterior (with gradients).

        Args:
            sample_shape: A `torch.Size` object specifying the sample shape. To
                draw `n` samples, set to `torch.Size([n])`. To draw `b` batches
                of `n` samples each, set to `torch.Size([b, n])`.
            base_samples: An (optional) Tensor of `N(0, I)` base samples of
                appropriate dimension, typically obtained from a `Sampler`.
                This is used for deterministic optimization.

        Returns:
            A `sample_shape x event`-dim Tensor of samples from the posterior.
        """
        pass

    def sample(self, sample_shape: Optional[torch.Size]=None, base_samples: Optional[Tensor]=None) ->Tensor:
        """Sample from the posterior (without gradients).

        This is a simple wrapper calling `rsample` using `with torch.no_grad()`.

        Args:
            sample_shape: A `torch.Size` object specifying the sample shape. To
                draw `n` samples, set to `torch.Size([n])`. To draw `b` batches
                of `n` samples each, set to `torch.Size([b, n])`.
            base_samples: An (optional) Tensor of `N(0, I)` base samples of
                appropriate dimension, typically obtained from a `Sampler` object.
                This is used for deterministic optimization.

        Returns:
            A `sample_shape x event_shape`-dim Tensor of samples from the posterior.
        """
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, base_samples=base_samples)


class MCSampler(Module, ABC):
    """Abstract base class for Samplers.

    Subclasses must implement the `_construct_base_samples` method.

    Attributes:
        resample: If `True`, re-draw samples in each `forward` evaluation -
            this results in stochastic acquisition functions (and thus should
            not be used with deterministic optimization algorithms).
        collapse_batch_dims: If True, collapse the t-batch dimensions of the
            produced samples to size 1. This is useful for preventing sampling
            variance across t-batches.

    Example:
        This method is usually not called directly, but via the sampler's
        `__call__` method:
        >>> posterior = model.posterior(test_X)
        >>> samples = sampler(posterior)
    """

    def forward(self, posterior: Posterior) ->Tensor:
        """Draws MC samples from the posterior.

        Args:
            posterior: The Posterior to sample from.

        Returns:
            The samples drawn from the posterior.
        """
        base_sample_shape = self._get_base_sample_shape(posterior=posterior)
        self._construct_base_samples(posterior=posterior, shape=base_sample_shape)
        samples = posterior.rsample(sample_shape=self.sample_shape, base_samples=self.base_samples)
        return samples

    def _get_base_sample_shape(self, posterior: Posterior) ->torch.Size:
        """Get the shape of the base samples.

        Args:
            posterior: The Posterior to sample from.

        Returns:
            The shape of the base samples expected by the posterior. If
            `collapse_batch_dims=True`, the t-batch dimensions of the base
            samples are collapsed to size 1. This is useful to prevent sampling
            variance across t-batches.
        """
        event_shape = posterior.event_shape
        if self.collapse_batch_dims:
            event_shape = torch.Size([(1) for _ in event_shape[:-2]]) + event_shape[-2:]
        return self.sample_shape + event_shape

    @property
    def sample_shape(self) ->torch.Size:
        """The shape of a single sample"""
        return self._sample_shape

    @abstractmethod
    def _construct_base_samples(self, posterior: Posterior, shape: torch.Size) ->None:
        """Generate base samples (if necessary).

        This function will generate a new set of base samples and register the
        `base_samples` buffer if one of the following is true:

         - `resample=True`
         - the MCSampler has no `base_samples` attribute.
         - `shape` is different than `self.base_samples.shape` (if
           `collapse_batch_dims=True`, then batch dimensions of will be
           automatically broadcasted as necessary)

        Args:
            posterior: The Posterior for which to generate base samples.
            shape: The shape of the base samples to construct.
        """
        pass


class AcquisitionObjective(Module, ABC):
    """Abstract base class for objectives."""
    ...


class BotorchError(Exception):
    """Base botorch exception."""
    pass


class BotorchTensorDimensionError(BotorchError):
    """Exception raised when a tensor violates a botorch convention."""
    pass


class UnsupportedError(BotorchError):
    """Currently unsupported feature."""
    pass


def t_batch_mode_transform(expected_q: Optional[int]=None) ->Callable[[Callable[[Any, Tensor], Any]], Callable[[Any, Tensor], Any]]:
    """Factory for decorators taking a t-batched `X` tensor.

    This method creates decorators for instance methods to transform an input tensor
    `X` to t-batch mode (i.e. with at least 3 dimensions). This assumes the tensor
    has a q-batch dimension. The decorator also checks the q-batch size if `expected_q`
    is provided.

    Args:
        expected_q: The expected q-batch size of X. If specified, this will raise an
            AssertitionError if X's q-batch size does not equal expected_q.

    Returns:
        The decorated instance method.

    Example:
        >>> class ExampleClass:
        >>>     @t_batch_mode_transform(expected_q=1)
        >>>     def single_q_method(self, X):
        >>>         ...
        >>>
        >>>     @t_batch_mode_transform()
        >>>     def arbitrary_q_method(self, X):
        >>>         ...
    """

    def decorator(method: Callable[[Any, Tensor], Any]) ->Callable[[Any, Tensor], Any]:

        @wraps(method)
        def decorated(cls: Any, X: Tensor) ->Any:
            if X.dim() < 2:
                raise ValueError(f'{type(cls).__name__} requires X to have at least 2 dimensions, but received X with only {X.dim()} dimensions.')
            elif expected_q is not None and X.shape[-2] != expected_q:
                raise AssertionError(f'Expected X to be `batch_shape x q={expected_q} x d`, but got X with shape {X.shape}.')
            X = X if X.dim() > 2 else X.unsqueeze(0)
            return method(cls, X)
        return decorated
    return decorator


def _construct_dist(means: Tensor, sigmas: Tensor, inds: Tensor) ->Normal:
    mean = means.index_select(dim=-1, index=inds)
    sigma = sigmas.index_select(dim=-1, index=inds)
    return Normal(loc=mean, scale=sigma)


def convert_to_target_pre_hook(module, *args):
    """Pre-hook for automatically calling `.to(X)` on module prior to `forward`"""
    module


class BotorchTensorDimensionWarning(BotorchWarning):
    """Warning raised when a tensor possibly violates a botorch convention."""
    pass


def add_output_dim(X: Tensor, original_batch_shape: torch.Size) ->Tuple[Tensor, int]:
    """Insert the output dimension at the correct location.

    The trailing batch dimensions of X must match the original batch dimensions
    of the training inputs, but can also include extra batch dimensions.

    Args:
        X: A `(new_batch_shape) x (original_batch_shape) x n x d` tensor of
            features.
        original_batch_shape: the batch shape of the model's training inputs.

    Returns:
        2-element tuple containing

        - A `(new_batch_shape) x (original_batch_shape) x m x n x d` tensor of
            features.
        - The index corresponding to the output dimension.
    """
    X_batch_shape = X.shape[:-2]
    if len(X_batch_shape) > 0 and len(original_batch_shape) > 0:
        error_msg = 'The trailing batch dimensions of X must match the trailing batch dimensions of the training inputs.'
        _mul_broadcast_shape(X_batch_shape, original_batch_shape, error_msg=error_msg)
    X = X.unsqueeze(-3)
    output_dim_idx = max(len(original_batch_shape), len(X_batch_shape))
    return X, output_dim_idx


def mod_batch_shape(module: Module, names: List[str], b: int) ->None:
    """Recursive helper to modify gpytorch modules' batch shape attribute.

    Modifies the module in-place.

    Args:
        module: The module to be modified.
        names: The list of names to access the attribute. If the full name of
            the module is `"module.sub_module.leaf_module"`, this will be
            `["sub_module", "leaf_module"]`.
        b: The new size of the last element of the module's `batch_shape`
            attribute.
    """
    if len(names) == 0:
        return
    m = getattr(module, names[0])
    if len(names) == 1 and hasattr(m, 'batch_shape') and len(m.batch_shape) > 0:
        m.batch_shape = m.batch_shape[:-1] + torch.Size([b] if b > 0 else [])
    else:
        mod_batch_shape(module=m, names=names[1:], b=b)


def multioutput_to_batch_mode_transform(train_X: Tensor, train_Y: Tensor, num_outputs: int, train_Yvar: Optional[Tensor]=None) ->Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Transforms training inputs for a multi-output model.

    Used for multi-output models that internally are represented by a
    batched single output model, where each output is modeled as an
    independent batch.

    Args:
        train_X: A `n x d` or `input_batch_shape x n x d` (batch mode) tensor of
            training features.
        train_Y: A `n x m` or `target_batch_shape x n x m` (batch mode) tensor of
            training observations.
        num_outputs: number of outputs
        train_Yvar: A `n x m` or `target_batch_shape x n x m` tensor of observed
            measurement noise.

    Returns:
        3-element tuple containing

        - A `input_batch_shape x m x n x d` tensor of training features.
        - A `target_batch_shape x m x n` tensor of training observations.
        - A `target_batch_shape x m x n` tensor observed measurement noise.
    """
    train_Y = train_Y.transpose(-1, -2)
    train_X = train_X.unsqueeze(-3).expand(train_X.shape[:-2] + torch.Size([num_outputs]) + train_X.shape[-2:])
    if train_Yvar is not None:
        train_Yvar = train_Yvar.transpose(-1, -2)
    return train_X, train_Y, train_Yvar


class OutcomeTransform(Module, ABC):
    """Abstract base class for outcome transforms."""

    @abstractmethod
    def forward(self, Y: Tensor, Yvar: Optional[Tensor]=None) ->Tuple[Tensor, Optional[Tensor]]:
        """Transform the outcomes in a model's training targets

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        pass

    def untransform(self, Y: Tensor, Yvar: Optional[Tensor]=None) ->Tuple[Tensor, Optional[Tensor]]:
        """Un-transform previously transformed outcomes

        Args:
            Y: A `batch_shape x n x m`-dim tensor of transfomred training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of transformed observation
                noises associated with the training targets (if applicable).

        Returns:
            A two-tuple with the un-transformed outcomes:

            - The un-transformed outcome observations.
            - The un-transformed observation noise (if applicable).
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not implement the `untransform` method')

    def untransform_posterior(self, posterior: Posterior) ->Posterior:
        """Un-transform a posterior

        Args:
            posterior: A posterior in the transformed space.

        Returns:
            The un-transformed posterior.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not implement the `untransform_posterior` method')


class InputDataError(BotorchError):
    """Exception raised when input data does not comply with conventions."""
    pass


class InputDataWarning(BotorchWarning):
    """Warning raised when input data does not comply with conventions."""
    pass


def check_min_max_scaling(X: Tensor, strict: bool=False, atol: float=0.01, raise_on_fail: bool=False) ->None:
    """Check that tensor is normalized to the unit cube.

    Args:
        X: A `batch_shape x n x d` input tensor. Typically the training inputs
            of a model.
        strict: If True, require `X` to be scaled to the unit cube (rather than
            just to be contained within the unit cube).
        atol: The tolerance for the boundary check. Only used if `strict=True`.
        raise_on_fail: If True, raise an exception instead of a warning.
    """
    with torch.no_grad():
        Xmin, Xmax = torch.min(X, dim=-1)[0], torch.max(X, dim=-1)[0]
        msg = None
        if strict and max(torch.abs(Xmin).max(), torch.abs(Xmax - 1).max()) > atol:
            msg = 'scaled'
        if torch.any(Xmin < -atol) or torch.any(Xmax > 1 + atol):
            msg = 'contained'
        if msg is not None:
            msg = f'Input data is not {msg} to the unit cube. Please consider min-max scaling the input data.'
            if raise_on_fail:
                raise InputDataError(msg)
            warnings.warn(msg, InputDataWarning)


def check_no_nans(Z: Tensor) ->None:
    """Check that tensor does not contain NaN values.

    Raises an InputDataError if `Z` contains NaN values.

    Args:
        Z: The input tensor.
    """
    if torch.any(torch.isnan(Z)).item():
        raise InputDataError('Input data contains NaN values.')


def check_standardization(Y: Tensor, atol_mean: float=0.01, atol_std: float=0.01, raise_on_fail: bool=False) ->None:
    """Check that tensor is standardized (zero mean, unit variance).

    Args:
        Y: The input tensor of shape `batch_shape x n x m`. Typically the
            train targets of a model. Standardization is checked across the
            `n`-dimension.
        atol_mean: The tolerance for the mean check.
        atol_std: The tolerance for the std check.
        raise_on_fail: If True, raise an exception instead of a warning.
    """
    with torch.no_grad():
        Ymean, Ystd = torch.mean(Y, dim=-2), torch.std(Y, dim=-2)
        if torch.abs(Ymean).max() > atol_mean or torch.abs(Ystd - 1).max() > atol_std:
            msg = 'Input data is not standardized. Please consider scaling the input to zero mean and unit variance.'
            if raise_on_fail:
                raise InputDataError(msg)
            warnings.warn(msg, InputDataWarning)


def validate_input_scaling(train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor]=None, raise_on_fail: bool=False) ->None:
    """Helper function to validate input data to models.

    Args:
        train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of
            training features.
        train_Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
            training observations.
        train_Yvar: A `batch_shape x n x m` or `batch_shape x n x m` (batch mode)
            tensor of observed measurement noise.
        raise_on_fail: If True, raise an error instead of emitting a warning
            (only for normalization/standardization checks, an error is always
            raised if NaN values are present).

    This function is typically called inside the constructor of standard BoTorch
    models. It validates the following:
    (i) none of the inputs contain NaN values
    (ii) the training data (`train_X`) is normalized to the unit cube
    (iii) the training targets (`train_Y`) are standardized (zero mean, unit var)
    No checks (other than the NaN check) are performed for observed variances
    (`train_Yvar`) at this point.
    """
    if settings.validate_input_scaling.off():
        return
    check_no_nans(train_X)
    check_no_nans(train_Y)
    if train_Yvar is not None:
        check_no_nans(train_Yvar)
        if torch.any(train_Yvar < 0):
            raise InputDataError('Input data contains negative variances.')
    check_min_max_scaling(X=train_X, raise_on_fail=raise_on_fail)
    check_standardization(Y=train_Y, raise_on_fail=raise_on_fail)


class NormalQMCEngine:
    """Engine for qMC sampling from a Multivariate Normal `N(0, I_d)`.

    By default, this implementation uses Box-Muller transformed Sobol samples
    following pg. 123 in [Pages2018numprob]_. To use the inverse transform
    instead, set `inv_transform=True`.

    Example:
        >>> engine = NormalQMCEngine(3)
        >>> samples = engine.draw(10)
    """

    def __init__(self, d: int, seed: Optional[int]=None, inv_transform: bool=False) ->None:
        """Engine for drawing qMC samples from a multivariate normal `N(0, I_d)`.

        Args:
            d: The dimension of the samples.
            seed: The seed with which to seed the random number generator of the
                underlying SobolEngine.
            inv_transform: If True, use inverse transform instead of Box-Muller.
        """
        self._d = d
        self._seed = seed
        self._inv_transform = inv_transform
        if inv_transform:
            sobol_dim = d
        else:
            sobol_dim = 2 * math.ceil(d / 2)
        self._sobol_engine = SobolEngine(dimension=sobol_dim, scramble=True, seed=seed)

    def draw(self, n: int=1, out: Optional[Tensor]=None, dtype: torch.dtype=torch.float) ->Optional[Tensor]:
        """Draw `n` qMC samples from the standard Normal.

        Args:
            n: The number of samples to draw.
            out: An option output tensor. If provided, draws are put into this
                tensor, and the function returns None.
            dtype: The desired torch data type (ignored if `out` is provided).

        Returns:
            A `n x d` tensor of samples if `out=None` and `None` otherwise.
        """
        samples = self._sobol_engine.draw(n, dtype=dtype)
        if self._inv_transform:
            v = 0.5 + (1 - torch.finfo(samples.dtype).eps) * (samples - 0.5)
            samples_tf = torch.erfinv(2 * v - 1) * math.sqrt(2)
        else:
            even = torch.arange(0, samples.shape[-1], 2)
            Rs = (-2 * torch.log(samples[:, (even)])).sqrt()
            thetas = 2 * math.pi * samples[:, (1 + even)]
            cos = torch.cos(thetas)
            sin = torch.sin(thetas)
            samples_tf = torch.stack([Rs * cos, Rs * sin], -1).reshape(n, -1)
            samples_tf = samples_tf[:, :self._d]
        if out is None:
            return samples_tf
        else:
            out.copy_(samples_tf)


def draw_sobol_normal_samples(d: int, n: int, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None, seed: Optional[int]=None) ->Tensor:
    """Draw qMC samples from a multi-variate standard normal N(0, I_d)

    A primary use-case for this functionality is to compute an QMC average
    of f(X) over X where each element of X is drawn N(0, 1).

    Args:
        d: The dimension of the normal distribution.
        n: The number of samples to return.
        device: The torch device.
        dtype:  The torch dtype.
        seed: The seed used for initializing Owen scrambling. If None (default),
            use a random seed.

    Returns:
        A tensor of qMC standard normal samples with dimension `n x d` with device
        and dtype specified by the input.

    Example:
        >>> samples = draw_sobol_normal_samples(2, 10)
    """
    normal_qmc_engine = NormalQMCEngine(d=d, seed=seed, inv_transform=True)
    samples = normal_qmc_engine.draw(n, dtype=torch.float if dtype is None else dtype)
    return samples


class SobolQMCNormalSampler(MCSampler):
    """Sampler for quasi-MC base samples using Sobol sequences.

    Example:
        >>> sampler = SobolQMCNormalSampler(1000, seed=1234)
        >>> posterior = model.posterior(test_X)
        >>> samples = sampler(posterior)
    """

    def __init__(self, num_samples: int, resample: bool=False, seed: Optional[int]=None, collapse_batch_dims: bool=True) ->None:
        """Sampler for quasi-MC base samples using Sobol sequences.

        Args:
            num_samples: The number of samples to use.
            resample: If `True`, re-draw samples in each `forward` evaluation -
                this results in stochastic acquisition functions (and thus should
                not be used with deterministic optimization algorithms).
            seed: The seed for the RNG. If omitted, use a random seed.
            collapse_batch_dims: If True, collapse the t-batch dimensions to
                size 1. This is useful for preventing sampling variance across
                t-batches.
        """
        super().__init__()
        self._sample_shape = torch.Size([num_samples])
        self.collapse_batch_dims = collapse_batch_dims
        self.resample = resample
        self.seed = seed if seed is not None else torch.randint(0, 1000000, (1,)).item()

    def _construct_base_samples(self, posterior: Posterior, shape: torch.Size) ->None:
        """Generate quasi-random Normal base samples (if necessary).

        This function will generate a new set of base samples and set the
        `base_samples` buffer if one of the following is true:

        - `resample=True`
        - the MCSampler has no `base_samples` attribute.
        - `shape` is different than `self.base_samples.shape` (if
          `collapse_batch_dims=True`, then batch dimensions of will be
          automatically broadcasted as necessary)

        Args:
            posterior: The Posterior for which to generate base samples.
            shape: The shape of the base samples to construct.
        """
        if self.resample or not hasattr(self, 'base_samples') or self.base_samples.shape[-2:] != shape[-2:] or not self.collapse_batch_dims and shape != self.base_samples.shape:
            output_dim = shape[-2:].numel()
            if output_dim > SobolEngine.MAXDIM:
                raise UnsupportedError(f'SobolQMCSampler only supports dimensions `q * o <= {SobolEngine.MAXDIM}`. Requested: {output_dim}')
            base_samples = draw_sobol_normal_samples(d=output_dim, n=shape[:-2].numel(), device=posterior.device, dtype=posterior.dtype, seed=self.seed)
            self.seed += 1
            base_samples = base_samples.view(shape)
            self.register_buffer('base_samples', base_samples)
        elif self.collapse_batch_dims and shape != posterior.event_shape:
            self.base_samples = self.base_samples.view(shape)
        if self.base_samples.device != posterior.device:
            self
        if self.base_samples.dtype != posterior.dtype:
            self


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


class GenericCostAwareUtility(CostAwareUtility):
    """Generic cost-aware utility wrapping a callable."""

    def __init__(self, cost: Callable[[Tensor, Tensor], Tensor]) ->None:
        """Generic cost-aware utility wrapping a callable.

        Args:
            cost: A callable mapping a `batch_shape x q x d'`-dim candidate set
                to a `batch_shape`-dim tensor of costs
        """
        super().__init__()
        self._cost_callable: Callable[[Tensor, Tensor], Tensor] = cost

    def forward(self, X: Tensor, deltas: Tensor, **kwargs: Any) ->Tensor:
        """Evaluate the cost function on the candidates and improvements.

        Args:
            X: A `batch_shape x q x d'`-dim Tensor of with `q` `d`-dim design
                points for each t-batch.
            deltas: A `num_fantasies x batch_shape`-dim Tensor of `num_fantasy`
                samples from the marginal improvement in utility over the
                current state at `X` for each t-batch.

        Returns:
            A `num_fantasies x batch_shape`-dim Tensor of cost-weighted utilities.
        """
        return self._cost_callable(X, deltas)


class CostAwareWarning(BotorchWarning):
    """Warning raised in the context of cost-aware acquisition strategies."""
    pass


class MCAcquisitionObjective(AcquisitionObjective):
    """Abstract base class for MC-based objectives."""

    @abstractmethod
    def forward(self, samples: Tensor) ->Tensor:
        """Evaluate the objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of
                samples from a model posterior.

        Returns:
            Tensor: A `sample_shape x batch_shape x q`-dim Tensor of objective
            values (assuming maximization).

        This method is usually not called directly, but via the objectives

        Example:
            >>> # `__call__` method:
            >>> samples = sampler(posterior)
            >>> outcome = mc_obj(samples)
        """
        pass


class IdentityMCObjective(MCAcquisitionObjective):
    """Trivial objective extracting the last dimension.

    Example:
        >>> identity_objective = IdentityMCObjective()
        >>> samples = sampler(posterior)
        >>> objective = identity_objective(samples)
    """

    def forward(self, samples: Tensor) ->Tensor:
        return samples.squeeze(-1)


def match_batch_shape(X: Tensor, Y: Tensor) ->Tensor:
    """Matches the batch dimension of a tensor to that of another tensor.

    Args:
        X: A `batch_shape_X x q x d` tensor, whose batch dimensions that
            correspond to batch dimensions of `Y` are to be matched to those
            (if compatible).
        Y: A `batch_shape_Y x q' x d` tensor.

    Returns:
        A `batch_shape_Y x q x d` tensor containing the data of `X` expanded to
        the batch dimensions of `Y` (if compatible). For instance, if `X` is
        `b'' x b' x q x d` and `Y` is `b x q x d`, then the returned tensor is
        `b'' x b x q x d`.

    Example:
        >>> X = torch.rand(2, 1, 5, 3)
        >>> Y = torch.rand(2, 6, 4, 3)
        >>> X_matched = match_batch_shape(X, Y)
        >>> X_matched.shape
        torch.Size([2, 6, 5, 3])

    """
    return X.expand(X.shape[:-Y.dim()] + Y.shape[:-2] + X.shape[-2:])


def concatenate_pending_points(method: Callable[[Any, Tensor], Any]) ->Callable[[Any, Tensor], Any]:
    """Decorator concatenating X_pending into an acquisition function's argument.

    This decorator works on the `forward` method of acquisition functions taking
    a tensor `X` as the argument. If the acquisition function has an `X_pending`
    attribute (that is not `None`), this is concatenated into the input `X`,
    appropriately expanding the pending points to match the batch shape of `X`.

    Example:
        >>> class ExampleAcquisitionFunction:
        >>>     @concatenate_pending_points
        >>>     @t_batch_mode_transform()
        >>>     def forward(self, X):
        >>>         ...
    """

    @wraps(method)
    def decorated(cls: Any, X: Tensor, **kwargs: Any) ->Any:
        if cls.X_pending is not None:
            X = torch.cat([X, match_batch_shape(cls.X_pending, X)], dim=-2)
        return method(cls, X, **kwargs)
    return decorated


def _split_fantasy_points(X: Tensor, n_f: int) ->Tuple[Tensor, Tensor]:
    """Split a one-shot optimization input into actual and fantasy points

    Args:
        X: A `batch_shape x (q + n_f) x d`-dim tensor of actual and fantasy
            points

    Returns:
        2-element tuple containing

        - A `batch_shape x q x d`-dim tensor `X_actual` of input candidates.
        - A `n_f x batch_shape x 1 x d`-dim tensor `X_fantasies` of fantasy
            points, where `X_fantasies[i, batch_idx]` is the i-th fantasy point
            associated with the batch indexed by `batch_idx`.
    """
    if n_f > X.size(-2):
        raise ValueError(f'n_f ({n_f}) must be less than the q-batch dimension of X ({X.size(-2)})')
    split_sizes = [X.size(-2) - n_f, n_f]
    X_actual, X_fantasies = torch.split(X, split_sizes, dim=-2)
    X_fantasies = X_fantasies.permute(-2, *range(X_fantasies.dim() - 2), -1)
    X_fantasies = X_fantasies.unsqueeze(dim=-2)
    return X_actual, X_fantasies


class IIDNormalSampler(MCSampler):
    """Sampler for MC base samples using iid N(0,1) samples.

    Example:
        >>> sampler = IIDNormalSampler(1000, seed=1234)
        >>> posterior = model.posterior(test_X)
        >>> samples = sampler(posterior)
    """

    def __init__(self, num_samples: int, resample: bool=False, seed: Optional[int]=None, collapse_batch_dims: bool=True) ->None:
        """Sampler for MC base samples using iid `N(0,1)` samples.

        Args:
            num_samples: The number of samples to use.
            resample: If `True`, re-draw samples in each `forward` evaluation -
                this results in stochastic acquisition functions (and thus should
                not be used with deterministic optimization algorithms).
            seed: The seed for the RNG. If omitted, use a random seed.
            collapse_batch_dims: If True, collapse the t-batch dimensions to
                size 1. This is useful for preventing sampling variance across
                t-batches.
        """
        super().__init__()
        self._sample_shape = torch.Size([num_samples])
        self.collapse_batch_dims = collapse_batch_dims
        self.resample = resample
        self.seed = seed if seed is not None else torch.randint(0, 1000000, (1,)).item()

    def _construct_base_samples(self, posterior: Posterior, shape: torch.Size) ->None:
        """Generate iid `N(0,1)` base samples (if necessary).

        This function will generate a new set of base samples and set the
        `base_samples` buffer if one of the following is true:

        - `resample=True`
        - the MCSampler has no `base_samples` attribute.
        - `shape` is different than `self.base_samples.shape` (if
            `collapse_batch_dims=True`, then batch dimensions of will be
            automatically broadcasted as necessary)

        Args:
            posterior: The Posterior for which to generate base samples.
            shape: The shape of the base samples to construct.
        """
        if self.resample or not hasattr(self, 'base_samples') or self.base_samples.shape[-2:] != shape[-2:] or not self.collapse_batch_dims and shape != self.base_samples.shape:
            with manual_seed(seed=self.seed):
                base_samples = torch.randn(shape, device=posterior.device, dtype=posterior.dtype)
            self.seed += 1
            self.register_buffer('base_samples', base_samples)
        elif self.collapse_batch_dims and shape != self.base_samples.shape:
            self.base_samples = self.base_samples.view(shape)
        if self.base_samples.device != posterior.device:
            self
        if self.base_samples.dtype != posterior.dtype:
            self


class SamplingWarning(BotorchWarning):
    """Sampling related warnings."""
    pass


class LinearMCObjective(MCAcquisitionObjective):
    """Linear objective constructed from a weight tensor.

    For input `samples` and `mc_obj = LinearMCObjective(weights)`, this produces
    `mc_obj(samples) = sum_{i} weights[i] * samples[..., i]`

    Example:
        Example for a model with two outcomes:

        >>> weights = torch.tensor([0.75, 0.25])
        >>> linear_objective = LinearMCObjective(weights)
        >>> samples = sampler(posterior)
        >>> objective = linear_objective(samples)
    """

    def __init__(self, weights: Tensor) ->None:
        """Linear Objective.

        Args:
            weights: A one-dimensional tensor with `m` elements representing the
                linear weights on the outputs.
        """
        super().__init__()
        if weights.dim() != 1:
            raise ValueError('weights must be a one-dimensional tensor.')
        self.register_buffer('weights', weights)

    def forward(self, samples: Tensor) ->Tensor:
        """Evaluate the linear objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim tensors of
                samples from a model posterior.

        Returns:
            A `sample_shape x batch_shape x q`-dim tensor of objective values.
        """
        if samples.shape[-1] != self.weights.shape[-1]:
            raise RuntimeError('Output shape of samples not equal to that of weights')
        return torch.einsum('...m, m', [samples, self.weights])


class GenericMCObjective(MCAcquisitionObjective):
    """Objective generated from a generic callable.

    Allows to construct arbitrary MC-objective functions from a generic
    callable. In order to be able to use gradient-based acquisition function
    optimization it should be possible to backpropagate through the callable.

    Example:
        >>> generic_objective = GenericMCObjective(lambda Y: torch.sqrt(Y).sum(dim=-1))
        >>> samples = sampler(posterior)
        >>> objective = generic_objective(samples)
    """

    def __init__(self, objective: Callable[[Tensor], Tensor]) ->None:
        """Objective generated from a generic callable.

        Args:
            objective: A callable mapping a `sample_shape x batch-shape x q x m`-
                dim Tensor to a `sample_shape x batch-shape x q`-dim Tensor of
                objective values.
        """
        super().__init__()
        self.objective = objective

    def forward(self, samples: Tensor) ->Tensor:
        """Evaluate the feasibility-weigthed objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of
                samples from a model posterior.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of objective values
            weighted by feasibility (assuming maximization).
        """
        return self.objective(samples)


def soft_eval_constraint(lhs: Tensor, eta: float=0.001) ->Tensor:
    """Element-wise evaluation of a constraint in a 'soft' fashion

    `value(x) = 1 / (1 + exp(x / eta))`

    Args:
        lhs: The left hand side of the constraint `lhs <= 0`.
        eta: The temperature parameter of the softmax function. As eta
            grows larger, this approximates the Heaviside step function.

    Returns:
        Element-wise 'soft' feasibility indicator of the same shape as `lhs`.
        For each element `x`, `value(x) -> 0` as `x` becomes positive, and
        `value(x) -> 1` as x becomes negative.
    """
    if eta <= 0:
        raise ValueError('eta must be positive')
    return torch.sigmoid(-lhs / eta)


def apply_constraints_nonnegative_soft(obj: Tensor, constraints: List[Callable[[Tensor], Tensor]], samples: Tensor, eta: float) ->Tensor:
    """Applies constraints to a non-negative objective.

    This function uses a sigmoid approximation to an indicator function for
    each constraint.

    Args:
        obj: A `n_samples x b x q` Tensor of objective values.
        constraints: A list of callables, each mapping a Tensor of size `b x q x m`
            to a Tensor of size `b x q`, where negative values imply feasibility.
            This callable must support broadcasting. Only relevant for multi-
            output models (`m` > 1).
        samples: A `b x q x m` Tensor of samples drawn from the posterior.
        eta: The temperature parameter for the sigmoid function.

    Returns:
        A `n_samples x b x q`-dim tensor of feasibility-weighted objectives.
    """
    obj = obj.clamp_min(0)
    for constraint in constraints:
        obj = obj.mul(soft_eval_constraint(constraint(samples), eta=eta))
    return obj


def apply_constraints(obj: Tensor, constraints: List[Callable[[Tensor], Tensor]], samples: Tensor, infeasible_cost: float, eta: float=0.001) ->Tensor:
    """Apply constraints using an infeasible_cost `M` for negative objectives.

    This allows feasibility-weighting an objective for the case where the
    objective can be negative by usingthe following strategy:
    (1) add `M` to make obj nonnegative
    (2) apply constraints using the sigmoid approximation
    (3) shift by `-M`

    Args:
        obj: A `n_samples x b x q` Tensor of objective values.
        constraints: A list of callables, each mapping a Tensor of size `b x q x m`
            to a Tensor of size `b x q`, where negative values imply feasibility.
            This callable must support broadcasting. Only relevant for multi-
            output models (`m` > 1).
        samples: A `b x q x m` Tensor of samples drawn from the posterior.
        infeasible_cost: The infeasible value.
        eta: The temperature parameter of the sigmoid function.

    Returns:
        A `n_samples x b x q`-dim tensor of feasibility-weighted objectives.
    """
    obj = obj.add(infeasible_cost)
    obj = apply_constraints_nonnegative_soft(obj=obj, constraints=constraints, samples=samples, eta=eta)
    return obj.add(-infeasible_cost)


class ConstrainedMCObjective(GenericMCObjective):
    """Feasibility-weighted objective.

    An Objective allowing to maximize some scalable objective on the model
    outputs subject to a number of constraints. Constraint feasibilty is
    approximated by a sigmoid function.

    `mc_acq(X) = objective(X) * prod_i (1  - sigmoid(constraint_i(X)))`
    TODO: Document functional form exactly.

    See `botorch.utils.objective.apply_constraints` for details on the constarint
    handling.

    Example:
        >>> bound = 0.0
        >>> objective = lambda Y: Y[..., 0]
        >>> # apply non-negativity constraint on f(x)[1]
        >>> constraint = lambda Y: bound - Y[..., 1]
        >>> constrained_objective = ConstrainedMCObjective(objective, [constraint])
        >>> samples = sampler(posterior)
        >>> objective = constrained_objective(samples)
    """

    def __init__(self, objective: Callable[[Tensor], Tensor], constraints: List[Callable[[Tensor], Tensor]], infeasible_cost: float=0.0, eta: float=0.001) ->None:
        """Feasibility-weighted objective.

        Args:
            objective: A callable mapping a `sample_shape x batch-shape x q x m`-
                dim Tensor to a `sample_shape x batch-shape x q`-dim Tensor of
                objective values.
            constraints: A list of callables, each mapping a Tensor of dimension
                `sample_shape x batch-shape x q x m` to a Tensor of dimension
                `sample_shape x batch-shape x q`, where negative values imply
                feasibility.
            infeasible_cost: The cost of a design if all associated samples are
                infeasible.
            eta: The temperature parameter of the sigmoid function approximating
                the constraint.
        """
        super().__init__(objective=objective)
        self.constraints = constraints
        self.eta = eta
        self.register_buffer('infeasible_cost', torch.tensor(infeasible_cost))

    def forward(self, samples: Tensor) ->Tensor:
        """Evaluate the feasibility-weighted objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of
                samples from a model posterior.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of objective values
            weighted by feasibility (assuming maximization).
        """
        obj = super().forward(samples=samples)
        return apply_constraints(obj=obj, constraints=self.constraints, samples=samples, infeasible_cost=self.infeasible_cost, eta=self.eta)


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


def _flip_sub_unique(x: Tensor, k: int) ->Tensor:
    """Get the first k unique elements of a single-dimensional tensor, traversing the
    tensor from the back.

    Args:
        x: A single-dimensional tensor
        k: the number of elements to return

    Returns:
        A tensor with min(k, |x|) elements.

    Example:
        >>> x = torch.tensor([1, 6, 4, 3, 6, 3])
        >>> y = _flip_sub_unique(x, 3)  # tensor([3, 6, 4])
        >>> y = _flip_sub_unique(x, 4)  # tensor([3, 6, 4, 1])
        >>> y = _flip_sub_unique(x, 10)  # tensor([3, 6, 4, 1])

    NOTE: This should really be done in C++ to speed up the loop. Also, we would like
    to make this work for arbitrary batch shapes, I'm sure this can be sped up.
    """
    n = len(x)
    i = 0
    out = set()
    idcs = torch.empty(k, dtype=torch.long)
    for j, xi in enumerate(x.flip(0).tolist()):
        if xi not in out:
            out.add(xi)
            idcs[i] = n - 1 - j
            i += 1
        if len(out) >= k:
            break
    return x[idcs[:len(out)]]


def batched_multinomial(weights: Tensor, num_samples: int, replacement: bool=False, generator: Optional[torch.Generator]=None, out: Optional[Tensor]=None) ->LongTensor:
    """Sample from multinomial with an arbitrary number of batch dimensions.

    Args:
        weights: A `batch_shape x num_categories` tensor of weights. For each batch
            index `i, j, ...`, this functions samples from a multinomial with `input`
            `weights[i, j, ..., :]`. Note that the weights need not sum to one, but must
            be non-negative, finite and have a non-zero sum.
        num_samples: The number of samples to draw for each batch index. Must be smaller
            than `num_categories` if `replacement=False`.
        replacement: If True, samples are drawn with replacement.
        generator: A a pseudorandom number generator for sampling.
        out: The output tensor (optional). If provided, must be of size
            `batch_shape x num_samples`.

    Returns:
        A `batch_shape x num_samples` tensor of samples.

    This is a thin wrapper around `torch.multinomial` that allows weight (`input`)
    tensors with an arbitrary number of batch dimensions (`torch.multinomial` only
    allows a single batch dimension). The calling signature is the same as for
    `torch.multinomial`.

    Example:
        >>> weights = torch.rand(2, 3, 10)
        >>> samples = batched_multinomial(weights, 4)  # shape is 2 x 3 x 4
    """
    batch_shape, n_categories = weights.shape[:-1], weights.size(-1)
    flat_samples = torch.multinomial(input=weights.view(-1, n_categories), num_samples=num_samples, replacement=replacement, generator=generator, out=None if out is None else out.view(-1, num_samples))
    return flat_samples.view(*batch_shape, num_samples)


def standardize(Y: Tensor) ->Tensor:
    """Standardizes (zero mean, unit variance) a tensor by dim=-2.

    If the tensor is single-dimensional, simply standardizes the tensor.
    If for some batch index all elements are equal (of if there is only a single
    data point), this function will return 0 for that batch index.

    Args:
        Y: A `batch_shape x n x m`-dim tensor.

    Returns:
        The standardized `Y`.

    Example:
        >>> Y = torch.rand(4, 3)
        >>> Y_standardized = standardize(Y)
    """
    stddim = -1 if Y.dim() < 2 else -2
    Y_std = Y.std(dim=stddim, keepdim=True)
    Y_std = Y_std.where(Y_std >= 1e-09, torch.full_like(Y_std, 1.0))
    return (Y - Y.mean(dim=stddim, keepdim=True)) / Y_std


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


class Normalize(InputTransform):
    """Normalize the inputs to the unit cube.

    If no explicit bounds are provided this module is stateful: If in train mode,
    calling `forward` updates the module state (i.e. the normalizing bounds). If
    in eval mode, calling `forward` simply applies the normalization using the
    current module state.
    """

    def __init__(self, d: int, bounds: Optional[Tensor]=None, batch_shape: torch.Size=torch.Size()) ->None:
        """Normalize the inputs to the unit cube.

        Args:
            d: The dimension of the input space.
            bounds: If provided, use these bounds to normalize the inputs. If
                omitted, learn the bounds in train mode.
            batch_shape: The batch shape of the inputs (asssuming input tensors
                of shape `batch_shape x n x d`). If provided, perform individual
                normalization per batch, otherwise uses a single normalization.
        """
        super().__init__()
        if bounds is not None:
            if bounds.size(-1) != d:
                raise BotorchTensorDimensionError('Incompatible dimensions of provided bounds')
            mins = bounds[(...), 0:1, :]
            ranges = bounds[(...), 1:2, :] - mins
            self.learn_bounds = False
        else:
            mins = torch.zeros(*batch_shape, 1, d)
            ranges = torch.zeros(*batch_shape, 1, d)
            self.learn_bounds = True
        self.register_buffer('mins', mins)
        self.register_buffer('ranges', ranges)
        self._d = d

    def forward(self, X: Tensor) ->Tensor:
        """Normalize the inputs.

        If no explicit bounds are provided, this is stateful: In train mode,
        calling `forward` updates the module state (i.e. the normalizing bounds).
        In eval mode, calling `forward` simply applies the normalization using
        the current module state.

        Args:
            X: A `batch_shape x n x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of inputs normalized to the
            module's bounds.
        """
        if self.learn_bounds and self.training:
            if X.size(-1) != self.mins.size(-1):
                raise BotorchTensorDimensionError(f'Wrong input. dimension. Received {X.size(-1)}, expected {self.mins.size(-1)}')
            self.mins = X.min(dim=-2, keepdim=True)[0]
            self.ranges = X.max(dim=-2, keepdim=True)[0] - self.mins
        return (X - self.mins) / self.ranges

    def untransform(self, X: Tensor) ->Tensor:
        """Un-normalize the inputs.

        Args:
            X: A `batch_shape x n x d`-dim tensor of normalized inputs.

        Returns:
            A `batch_shape x n x d`-dim tensor of un-normalized inputs.
        """
        return self.mins + X * self.ranges

    @property
    def bounds(self) ->Tensor:
        """The bounds used for normalizing the inputs."""
        return torch.cat([self.mins, self.mins + self.ranges], dim=-2)


class ChainedOutcomeTransform(OutcomeTransform, ModuleDict):
    """An outcome transform representing the chaining of individual transforms"""

    def __init__(self, **transforms: OutcomeTransform) ->None:
        """Chaining of outcome transforms.

        Args:
            transforms: The transforms to chain. Internally, the names of the
                kwargs are used as the keys for accessing the individual
                transforms on the module.
        """
        super().__init__(OrderedDict(transforms))

    def forward(self, Y: Tensor, Yvar: Optional[Tensor]=None) ->Tuple[Tensor, Optional[Tensor]]:
        """Transform the outcomes in a model's training targets

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        for tf in self.values():
            Y, Yvar = tf.forward(Y, Yvar)
        return Y, Yvar

    def untransform(self, Y: Tensor, Yvar: Optional[Tensor]=None) ->Tuple[Tensor, Optional[Tensor]]:
        """Un-transform previously transformed outcomes

        Args:
            Y: A `batch_shape x n x m`-dim tensor of transfomred training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of transformed observation
                noises associated with the training targets (if applicable).

        Returns:
            A two-tuple with the un-transformed outcomes:

            - The un-transformed outcome observations.
            - The un-transformed observation noise (if applicable).
        """
        for tf in reversed(self.values()):
            Y, Yvar = tf.untransform(Y, Yvar)
        return Y, Yvar

    def untransform_posterior(self, posterior: Posterior) ->Posterior:
        """Un-transform a posterior

        Args:
            posterior: A posterior in the transformed space.

        Returns:
            The un-transformed posterior.
        """
        for tf in reversed(self.values()):
            posterior = tf.untransform_posterior(posterior)
        return posterior


class TransformedPosterior(Posterior):
    """An generic transformation of a posterior (implicitly represented)"""

    def __init__(self, posterior: Posterior, sample_transform: Callable[[Tensor], Tensor], mean_transform: Optional[Callable[[Tensor, Tensor], Tensor]]=None, variance_transform: Optional[Callable[[Tensor, Tensor], Tensor]]=None) ->None:
        """An implicitly represented transformed posterior

        Args:
            posterior: The posterior object to be transformed.
            sample_transform: A callable applying a sample-level transform to a
                `sample_shape x batch_shape x q x m`-dim tensor of samples from
                the original posterior, returning a tensor of samples of the
                same shape.
            mean_transform: A callable transforming a 2-tuple of mean and
                variance (both of shape `batch_shape x m x o`) of the original
                posterior to the mean of the transformed posterior.
            variance_transform: A callable transforming a 2-tuple of mean and
                variance (both of shape `batch_shape x m x o`) of the original
                posterior to a variance of the transformed posterior.
        """
        self._posterior = posterior
        self._sample_transform = sample_transform
        self._mean_transform = mean_transform
        self._variance_transform = variance_transform

    @property
    def device(self) ->torch.device:
        """The torch device of the posterior."""
        return self._posterior.device

    @property
    def dtype(self) ->torch.dtype:
        """The torch dtype of the posterior."""
        return self._posterior.dtype

    @property
    def event_shape(self) ->torch.Size:
        """The event shape (i.e. the shape of a single sample)."""
        return self._posterior.event_shape

    @property
    def mean(self) ->Tensor:
        """The mean of the posterior as a `batch_shape x n x m`-dim Tensor."""
        if self._mean_transform is None:
            raise NotImplementedError('No mean transform provided.')
        return self._mean_transform(self._posterior.mean, self._posterior.variance)

    @property
    def variance(self) ->Tensor:
        """The variance of the posterior as a `batch_shape x n x m`-dim Tensor."""
        if self._variance_transform is None:
            raise NotImplementedError('No variance transform provided.')
        return self._variance_transform(self._posterior.mean, self._posterior.variance)

    def rsample(self, sample_shape: Optional[torch.Size]=None, base_samples: Optional[Tensor]=None) ->Tensor:
        """Sample from the posterior (with gradients).

        Args:
            sample_shape: A `torch.Size` object specifying the sample shape. To
                draw `n` samples, set to `torch.Size([n])`. To draw `b` batches
                of `n` samples each, set to `torch.Size([b, n])`.
            base_samples: An (optional) Tensor of `N(0, I)` base samples of
                appropriate dimension, typically obtained from a `Sampler`.
                This is used for deterministic optimization.

        Returns:
            A `sample_shape x event`-dim Tensor of samples from the posterior.
        """
        samples = self._posterior.rsample(sample_shape=sample_shape, base_samples=base_samples)
        return self._sample_transform(samples)


def normalize_indices(indices: Optional[List[int]], d: int) ->Optional[List[int]]:
    """Normalize a list of indices to ensure that they are positive.

    Args:
        indices: A list of indices (may contain negative indices for indexing
            "from the back").
        d: The dimension of the tensor to index.

    Returns:
        A normalized list of indices such that each index is between `0` and
        `d-1`, or None if indices is None.
    """
    if indices is None:
        return indices
    normalized_indices = []
    for i in indices:
        if i < 0:
            i = i + d
        if i < 0 or i > d - 1:
            raise ValueError(f'Index {i} out of bounds for tensor or length {d}.')
        normalized_indices.append(i)
    return normalized_indices


class Standardize(OutcomeTransform):
    """Standardize outcomes (zero mean, unit variance).

    This module is stateful: If in train mode, calling forward updates the
    module state (i.e. the mean/std normalizing constants). If in eval mode,
    calling forward simply applies the standardization using the current module
    state.
    """

    def __init__(self, m: int, outputs: Optional[List[int]]=None, batch_shape: torch.Size=torch.Size(), min_stdv: float=1e-08) ->None:
        """Standardize outcomes (zero mean, unit variance).

        Args:
            m: The output dimension.
            outputs: Which of the outputs to standardize. If omitted, all
                outputs will be standardized.
            batch_shape: The batch_shape of the training targets.
            min_stddv: The minimum standard deviation for which to perform
                standardization (if lower, only de-mean the data).
        """
        super().__init__()
        self.register_buffer('means', torch.zeros(*batch_shape, 1, m))
        self.register_buffer('stdvs', torch.zeros(*batch_shape, 1, m))
        self._outputs = normalize_indices(outputs, d=m)
        self._m = m
        self._batch_shape = batch_shape
        self._min_stdv = min_stdv

    def forward(self, Y: Tensor, Yvar: Optional[Tensor]=None) ->Tuple[Tensor, Optional[Tensor]]:
        """Standardize outcomes.

        If the module is in train mode, this updates the module state (i.e. the
        mean/std normalizing constants). If the module is in eval mode, simply
        applies the normalization using the module state.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        if self.training:
            if Y.shape[:-2] != self._batch_shape:
                raise RuntimeError('wrong batch shape')
            if Y.size(-1) != self._m:
                raise RuntimeError('wrong output dimension')
            stdvs = Y.std(dim=-2, keepdim=True)
            stdvs = stdvs.where(stdvs >= self._min_stdv, torch.full_like(stdvs, 1.0))
            means = Y.mean(dim=-2, keepdim=True)
            if self._outputs is not None:
                unused = [i for i in range(self._m) if i not in self._outputs]
                means[..., unused] = 0.0
                stdvs[..., unused] = 1.0
            self.means = means
            self.stdvs = stdvs
            self._stdvs_sq = stdvs.pow(2)
        Y_tf = (Y - self.means) / self.stdvs
        Yvar_tf = Yvar / self._stdvs_sq if Yvar is not None else None
        return Y_tf, Yvar_tf

    def untransform(self, Y: Tensor, Yvar: Optional[Tensor]=None) ->Tuple[Tensor, Optional[Tensor]]:
        """Un-standardize outcomes.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of standardized targets.
            Yvar: A `batch_shape x n x m`-dim tensor of standardized observation
                noises associated with the targets (if applicable).

        Returns:
            A two-tuple with the un-standardized outcomes:

            - The un-standardized outcome observations.
            - The un-standardized observation noise (if applicable).
        """
        Y_utf = self.means + self.stdvs * Y
        Yvar_utf = self._stdvs_sq * Yvar if Yvar is not None else None
        return Y_utf, Yvar_utf

    def untransform_posterior(self, posterior: Posterior) ->Posterior:
        """Un-standardize the posterior.

        Args:
            posterior: A posterior in the standardized space.

        Returns:
            The un-standardized posterior. If the input posterior is a MVN,
            the transformed posterior is again an MVN.
        """
        if self._outputs is not None:
            raise NotImplementedError('Standardize does not yet support output selection for untransform_posterior')
        if not self._m == posterior.event_shape[-1]:
            raise RuntimeError(f'Incompatible output dimensions encountered for transform {self._m} and posterior {posterior.event_shape[-1]}')
        if not isinstance(posterior, GPyTorchPosterior):
            return TransformedPosterior(posterior=posterior, sample_transform=lambda s: self.means + self.stdvs * s, mean_transform=lambda m, v: self.means + self.stdvs * m, variance_transform=lambda m, v: self._stdvs_sq * v)
        mvn = posterior.mvn
        offset = self.means
        scale_fac = self.stdvs
        if not posterior._is_mt:
            mean_tf = offset.squeeze(-1) + scale_fac.squeeze(-1) * mvn.mean
            scale_fac = scale_fac.squeeze(-1).expand_as(mean_tf)
        else:
            mean_tf = offset + scale_fac * mvn.mean
            reps = mean_tf.shape[-2:].numel() // scale_fac.size(-1)
            scale_fac = scale_fac.squeeze(-2)
            if mvn._interleaved:
                scale_fac = scale_fac.repeat(*[(1) for _ in scale_fac.shape[:-1]], reps)
            else:
                scale_fac = torch.repeat_interleave(scale_fac, reps, dim=-1)
        if not mvn.islazy or mvn._MultivariateNormal__unbroadcasted_scale_tril is not None:
            covar_tf = CholLazyTensor(mvn.scale_tril * scale_fac.unsqueeze(-1))
        else:
            lcv = mvn.lazy_covariance_matrix
            scale_mat = DiagLazyTensor(scale_fac.expand(lcv.shape[:-1]))
            covar_tf = scale_mat @ lcv @ scale_mat
        kwargs = {'interleaved': mvn._interleaved} if posterior._is_mt else {}
        mvn_tf = mvn.__class__(mean=mean_tf, covariance_matrix=covar_tf, **kwargs)
        return GPyTorchPosterior(mvn_tf)


def norm_to_lognorm_mean(mu: Tensor, var: Tensor) ->Tensor:
    """Compute mean of a log-MVN from its MVN marginals

    Args:
        mu: A `batch_shape x n` mean vector of the Normal distribution.
        var: A `batch_shape x n` variance vectorof the Normal distribution.

    Returns:
        The `batch_shape x n` mean vector of the log-Normal distribution
    """
    return torch.exp(mu + 0.5 * var)


def norm_to_lognorm_variance(mu: Tensor, var: Tensor) ->Tensor:
    """Compute variance of a log-MVN from its MVN marginals

    Args:
        mu: A `batch_shape x n` mean vector of the Normal distribution.
        var: A `batch_shape x n` variance vectorof the Normal distribution.

    Returns:
        The `batch_shape x n` variance vector of the log-Normal distribution.
    """
    b = mu + 0.5 * var
    return (torch.exp(var) - 1) * torch.exp(2 * b)


class Log(OutcomeTransform):
    """Log-transform outcomes.

    Useful if the targets are modeled using a (multivariate) log-Normal
    distribution. This means that we can use a standard GP model on the
    log-transformed outcomes and un-transform the model posterior of that GP.
    """

    def __init__(self, outputs: Optional[List[int]]=None) ->None:
        """Log-transform outcomes.

        Args:
            outputs: Which of the outputs to log-transform. If omitted, all
                outputs will be standardized.
        """
        super().__init__()
        self._outputs = outputs

    def forward(self, Y: Tensor, Yvar: Optional[Tensor]=None) ->Tuple[Tensor, Optional[Tensor]]:
        """Log-transform outcomes.

        Args:
            Y: A `batch_shape x n x m`-dim tensor of training targets.
            Yvar: A `batch_shape x n x m`-dim tensor of observation noises
                associated with the training targets (if applicable).

        Returns:
            A two-tuple with the transformed outcomes:

            - The transformed outcome observations.
            - The transformed observation noise (if applicable).
        """
        Y_tf = torch.log(Y)
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_tf = torch.stack([(Y_tf[..., i] if i in outputs else Y[..., i]) for i in range(Y.size(-1))], dim=-1)
        if Yvar is not None:
            raise NotImplementedError('Log does not yet support transforming observation noise')
        return Y_tf, Yvar

    def untransform(self, Y: Tensor, Yvar: Optional[Tensor]=None) ->Tuple[Tensor, Optional[Tensor]]:
        """Un-transform log-transformed outcomes

        Args:
            Y: A `batch_shape x n x m`-dim tensor of log-transfomred targets.
            Yvar: A `batch_shape x n x m`-dim tensor of log- transformed
                observation noises associated with the training targets
                (if applicable).

        Returns:
            A two-tuple with the un-transformed outcomes:

            - The exponentiated outcome observations.
            - The exponentiated observation noise (if applicable).
        """
        Y_utf = torch.exp(Y)
        outputs = normalize_indices(self._outputs, d=Y.size(-1))
        if outputs is not None:
            Y_utf = torch.stack([(Y_utf[..., i] if i in outputs else Y[..., i]) for i in range(Y.size(-1))], dim=-1)
        if Yvar is not None:
            raise NotImplementedError('Log does not yet support transforming observation noise')
        return Y_utf, Yvar

    def untransform_posterior(self, posterior: Posterior) ->Posterior:
        """Un-transform the log-transformed posterior.

        Args:
            posterior: A posterior in the log-transformed space.

        Returns:
            The un-transformed posterior.
        """
        if self._outputs is not None:
            raise NotImplementedError('Log does not yet support output selection for untransform_posterior')
        return TransformedPosterior(posterior=posterior, sample_transform=torch.exp, mean_transform=norm_to_lognorm_mean, variance_transform=norm_to_lognorm_variance)


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


class ConstrainedBaseTestProblem(BaseTestProblem, ABC):
    """Base class for test functions with constraints.

    In addition to one or more objectives, a problem may have a number of outcome
    constraints of the form `c_i(x) >= 0` for `i=1, ..., n_c`.

    This base class provides common functionality for such problems.
    """
    num_constraints: int
    _check_grad_at_opt: bool = False

    def evaluate_slack(self, X: Tensor, noise: bool=True) ->Tensor:
        """Evaluate the constraint slack on a set of points.

        Constraints `i` is assumed to be feasible at `x` if the associated slack
        `c_i(x)` is positive. Zero slack means that the constraint is active. Negative
        slack means that the constraint is violated.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                constraint slacks: `c_1(X), ...., c_{n_c}(X)`.
            noise: If `True`, add observation noise to the slack as specified by
                `noise_std`.

        Returns:
            A `batch_shape x n_c`-dim tensor of constraint slack (where positive slack
                corresponds to the constraint being feasible).
        """
        cons = self.evaluate_slack_true(X=X)
        if noise and self.noise_std is not None:
            cons += self.noise_std * torch.randn_like(cons)
        return cons

    def is_feasible(self, X: Tensor, noise: bool=True) ->Tensor:
        """Evaluate whether the constraints are feasible on a set of points.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                constraints.
            noise: If `True`, add observation noise as specified by `noise_std`.

        Returns:
            A `batch_shape`-dim boolean tensor that is `True` iff all constraint
                slacks (potentially including observation noise) are positive.
        """
        return (self.evaluate_slack(X=X, noise=noise) >= 0.0).all(dim=-1)

    @abstractmethod
    def evaluate_slack_true(self, X: Tensor) ->Tensor:
        """Evaluate the constraint slack (w/o observation noise) on a set of points.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                constraint slacks: `c_1(X), ...., c_{n_c}(X)`.

        Returns:
            A `batch_shape x n_c`-dim tensor of constraint slack (where positive slack
                corresponds to the constraint being feasible).
        """
        pass


class SyntheticTestFunction(BaseTestProblem):
    """Base class for synthetic test functions."""
    _optimizers: List[Tuple[float, ...]]
    _optimal_value: float
    num_objectives: int = 1

    def __init__(self, noise_std: Optional[float]=None, negate: bool=False) ->None:
        """Base constructor for synthetic test functions.

        Arguments:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
        """
        super().__init__(noise_std=noise_std, negate=negate)
        if self._optimizers is not None:
            self.register_buffer('optimizers', torch.tensor(self._optimizers, dtype=torch.float))

    @property
    def optimal_value(self) ->float:
        """The global minimum (maximum if negate=True) of the function."""
        return -self._optimal_value if self.negate else self._optimal_value


class Ackley(SyntheticTestFunction):
    """Ackley test function.

    d-dimensional function (usually evaluated on `[-32.768, 32.768]^d`):

        f(x) = -A exp(-B sqrt(1/d sum_{i=1}^d x_i^2)) -
            exp(1/d sum_{i=1}^d cos(c x_i)) + A + exp(1)

    f has one minimizer for its global minimum at `z_1 = (0, 0, ..., 0)` with
    `f(z_1) = 0`.
    """
    _optimal_value = 0.0
    _check_grad_at_opt: bool = False

    def __init__(self, dim: int=2, noise_std: Optional[float]=None, negate: bool=False) ->None:
        self.dim = dim
        self._bounds = [(-32.768, 32.768) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)
        self.a = 20
        self.b = 0.2
        self.c = 2 * math.pi

    def evaluate_true(self, X: Tensor) ->Tensor:
        a, b, c = self.a, self.b, self.c
        part1 = -a * torch.exp(-b / math.sqrt(self.dim) * torch.norm(X, dim=-1))
        part2 = -torch.exp(torch.mean(torch.cos(c * X), dim=-1))
        return part1 + part2 + a + math.e


class Beale(SyntheticTestFunction):
    dim = 2
    _optimal_value = 0.0
    _bounds = [(-4.5, 4.5), (-4.5, 4.5)]
    _optimizers = [(3.0, 0.5)]

    def evaluate_true(self, X: Tensor) ->Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        part1 = (1.5 - x1 + x1 * x2) ** 2
        part2 = (2.25 - x1 + x1 * x2 ** 2) ** 2
        part3 = (2.625 - x1 + x1 * x2 ** 3) ** 2
        return part1 + part2 + part3


class Branin(SyntheticTestFunction):
    """Branin test function.

    Two-dimensional function (usually evaluated on `[-5, 10] x [0, 15]`):

        B(x) = (x_2 - b x_1^2 + c x_1 - r)^2 + 10 (1-t) cos(x_1) + 10

    Here `b`, `c`, `r` and `t` are constants where `b = 5.1 / (4 * math.pi ** 2)`
    `c = 5 / math.pi`, `r = 6`, `t = 1 / (8 * math.pi)`
    B has 3 minimizers for its global minimum at `z_1 = (-pi, 12.275)`,
    `z_2 = (pi, 2.275)`, `z_3 = (9.42478, 2.475)` with `B(z_i) = 0.397887`.
    """
    dim = 2
    _bounds = [(-5.0, 10.0), (0.0, 15.0)]
    _optimal_value = 0.397887
    _optimizers = [(-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)]

    def evaluate_true(self, X: Tensor) ->Tensor:
        t1 = X[..., 1] - 5.1 / (4 * math.pi ** 2) * X[..., 0] ** 2 + 5 / math.pi * X[..., 0] - 6
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X[..., 0])
        return t1 ** 2 + t2 + 10


class Bukin(SyntheticTestFunction):
    dim = 2
    _bounds = [(-15.0, -5.0), (-3.0, 3.0)]
    _optimal_value = 0.0
    _optimizers = [(-10.0, 1.0)]
    _check_grad_at_opt: bool = False

    def evaluate_true(self, X: Tensor) ->Tensor:
        part1 = 100.0 * torch.sqrt(torch.abs(X[..., 1] - 0.01 * X[..., 0] ** 2))
        part2 = 0.01 * torch.abs(X[..., 0] + 10.0)
        return part1 + part2


class Cosine8(SyntheticTestFunction):
    """Cosine Mixture test function.

    8-dimensional function (usually evaluated on `[-1, 1]^8`):

        f(x) = 0.1 sum_{i=1}^8 cos(5 pi x_i) - sum_{i=1}^8 x_i^2

    f has one maximizer for its global maximum at `z_1 = (0, 0, ..., 0)` with
    `f(z_1) = 0.8`
    """
    dim = 8
    _bounds = [(-1.0, 1.0) for _ in range(8)]
    _optimal_value = 0.8
    _optimizers = [tuple(0.0 for _ in range(8))]

    def evaluate_true(self, X: Tensor) ->Tensor:
        return torch.sum(0.1 * torch.cos(5 * math.pi * X) - X ** 2, dim=-1)


class DropWave(SyntheticTestFunction):
    dim = 2
    _bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    _optimal_value = -1.0
    _optimizers = [(0.0, 0.0)]
    _check_grad_at_opt = False

    def evaluate_true(self, X: Tensor) ->Tensor:
        norm = torch.norm(X, dim=-1)
        part1 = 1.0 + torch.cos(12.0 * norm)
        part2 = 0.5 * norm.pow(2) + 2.0
        return -part1 / part2


class DixonPrice(SyntheticTestFunction):
    _optimal_value = 0.0

    def __init__(self, dim=2, noise_std: Optional[float]=None, negate: bool=False) ->None:
        self.dim = dim
        self._bounds = [(-10.0, 10.0) for _ in range(self.dim)]
        self._optimizers = [tuple(math.pow(2.0, -(1.0 - 2.0 ** -(i - 1))) for i in range(1, self.dim + 1))]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) ->Tensor:
        d = self.dim
        part1 = (X[..., 0] - 1) ** 2
        i = X.new(range(2, d + 1))
        part2 = torch.sum(i * (2.0 * X[(...), 1:] ** 2 - X[(...), :-1]) ** 2, dim=1)
        return part1 + part2


class EggHolder(SyntheticTestFunction):
    """Eggholder test function.

    Two-dimensional function (usually evaluated on `[-512, 512]^2`):

        E(x) = (x_2 + 47) sin(R1(x)) - x_1 * sin(R2(x))

    where `R1(x) = sqrt(|x_2 + x_1 / 2 + 47|)`, `R2(x) = sqrt|x_1 - (x_2 + 47)|)`.
    """
    dim = 2
    _bounds = [(-512.0, 512.0), (-512.0, 512.0)]
    _optimal_value = -959.6407
    _optimizers = [(512.0, 404.2319)]
    _check_grad_at_opt: bool = False

    def evaluate_true(self, X: Tensor) ->Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        part1 = -(x2 + 47.0) * torch.sin(torch.sqrt(torch.abs(x2 + x1 / 2.0 + 47.0)))
        part2 = -x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47.0))))
        return part1 + part2


class Griewank(SyntheticTestFunction):
    _optimal_value = 0.0

    def __init__(self, dim=2, noise_std: Optional[float]=None, negate: bool=False) ->None:
        self.dim = dim
        self._bounds = [(-600.0, 600.0) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) ->Tensor:
        part1 = torch.sum(X ** 2 / 4000.0, dim=1)
        d = X.shape[1]
        part2 = -torch.prod(torch.cos(X / torch.sqrt(X.new(range(1, d + 1))).view(1, -1)))
        return part1 + part2 + 1.0


class Hartmann(SyntheticTestFunction):
    """Hartmann synthetic test function.

    Most commonly used is the six-dimensional version (typically evaluated on
    `[0, 1]^6`):

        H(x) = - sum_{i=1}^4 ALPHA_i exp( - sum_{j=1}^6 A_ij (x_j - P_ij)**2 )

    H has a 6 local minima and a global minimum at

        z = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)

    with `H(z) = -3.32237`.
    """

    def __init__(self, dim=6, noise_std: Optional[float]=None, negate: bool=False) ->None:
        if dim not in (3, 4, 6):
            raise ValueError(f'Hartmann with dim {dim} not defined')
        self.dim = dim
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        optvals = {(3): -3.86278, (6): -3.32237}
        optimizers = {(3): [(0.114614, 0.555649, 0.852547)], (6): [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)]}
        self._optimal_value = optvals.get(self.dim)
        self._optimizers = optimizers.get(self.dim)
        super().__init__(noise_std=noise_std, negate=negate)
        self.register_buffer('ALPHA', torch.tensor([1.0, 1.2, 3.0, 3.2]))
        if dim == 3:
            A = [[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]]
            P = [[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]]
        elif dim == 4:
            A = [[10, 3, 17, 3.5], [0.05, 10, 17, 0.1], [3, 3.5, 1.7, 10], [17, 8, 0.05, 10]]
            P = [[1312, 1696, 5569, 124], [2329, 4135, 8307, 3736], [2348, 1451, 3522, 2883], [4047, 8828, 8732, 5743]]
        elif dim == 6:
            A = [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]]
            P = [[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991], [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]]
        self.register_buffer('A', torch.tensor(A, dtype=torch.float))
        self.register_buffer('P', torch.tensor(P, dtype=torch.float))

    @property
    def optimal_value(self) ->float:
        if self.dim == 4:
            raise NotImplementedError()
        return super().optimal_value

    @property
    def optimizers(self) ->Tensor:
        if self.dim == 4:
            raise NotImplementedError()
        return super().optimizers

    def evaluate_true(self, X: Tensor) ->Tensor:
        self
        inner_sum = torch.sum(self.A * (X.unsqueeze(1) - 0.0001 * self.P) ** 2, dim=2)
        H = -torch.sum(self.ALPHA * torch.exp(-inner_sum), dim=1)
        if self.dim == 4:
            H = (1.1 + H) / 0.839
        return H


class HolderTable(SyntheticTestFunction):
    """Holder Table synthetic test function.

    Two-dimensional function (typically evaluated on `[0, 10] x [0, 10]`):

        `H(x) = - | sin(x_1) * cos(x_2) * exp(| 1 - ||x|| / pi | ) |`

    H has 4 global minima with `H(z_i) = -19.2085` at

        z_1 = ( 8.05502,  9.66459)
        z_2 = (-8.05502, -9.66459)
        z_3 = (-8.05502,  9.66459)
        z_4 = ( 8.05502, -9.66459)
    """
    dim = 2
    _bounds = [(-10.0, 10.0)]
    _optimal_value = -19.2085
    _optimizers = [(8.05502, 9.66459), (-8.05502, -9.66459), (-8.05502, 9.66459), (8.05502, -9.66459)]

    def evaluate_true(self, X: Tensor) ->Tensor:
        term = torch.abs(1 - torch.norm(X, dim=1) / math.pi)
        return -torch.abs(torch.sin(X[..., 0]) * torch.cos(X[..., 1]) * torch.exp(term))


class Levy(SyntheticTestFunction):
    """Levy synthetic test function.

    d-dimensional function (usually evaluated on `[-10, 10]^d`):

        f(x) = sin^2(pi w_1) +
            sum_{i=1}^{d-1} (w_i-1)^2 (1 + 10 sin^2(pi w_i + 1)) +
            (w_d - 1)^2 (1 + sin^2(2 pi w_d))

    where `w_i = 1 + (x_i - 1) / 4` for all `i`.

    f has one minimizer for its global minimum at `z_1 = (1, 1, ..., 1)` with
    `f(z_1) = 0`.
    """
    _optimal_value = 0.0

    def __init__(self, dim=2, noise_std: Optional[float]=None, negate: bool=False) ->None:
        self.dim = dim
        self._bounds = [(-10.0, 10.0) for _ in range(self.dim)]
        self._optimizers = [tuple(1.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) ->Tensor:
        w = 1.0 + (X - 1.0) / 4.0
        part1 = torch.sin(math.pi * w[..., 0]) ** 2
        part2 = torch.sum((w[(...), :-1] - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(math.pi * w[(...), :-1] + 1.0) ** 2), dim=1)
        part3 = (w[..., -1] - 1.0) ** 2 * (1.0 + torch.sin(2.0 * math.pi * w[..., -1]) ** 2)
        return part1 + part2 + part3


class Michalewicz(SyntheticTestFunction):
    """Michalewicz synthetic test function.

    d-dim function (usually evaluated on hypercube [0, pi]^d):

        M(x) = sum_{i=1}^d sin(x_i) (sin(i x_i^2 / pi)^20)
    """

    def __init__(self, dim=2, noise_std: Optional[float]=None, negate: bool=False) ->None:
        self.dim = dim
        self._bounds = [(0.0, math.pi) for _ in range(self.dim)]
        optvals = {(2): -1.80130341, (5): -4.687658, (10): -9.66015}
        optimizers = {(2): [(2.20290552, 1.57079633)]}
        self._optimal_value = optvals.get(self.dim)
        self._optimizers = optimizers.get(self.dim)
        super().__init__(noise_std=noise_std, negate=negate)
        self.register_buffer('i', torch.tensor(tuple(range(1, self.dim + 1)), dtype=torch.float))

    @property
    def optimizers(self) ->Tensor:
        if self.dim in (5, 10):
            raise NotImplementedError()
        return super().optimizers

    def evaluate_true(self, X: Tensor) ->Tensor:
        self
        m = 10
        return -torch.sum(torch.sin(X) * torch.sin(self.i * X ** 2 / math.pi) ** (2 * m), dim=-1)


class Powell(SyntheticTestFunction):
    _optimal_value = 0.0

    def __init__(self, dim=4, noise_std: Optional[float]=None, negate: bool=False) ->None:
        self.dim = dim
        self._bounds = [(-4.0, 5.0) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) ->Tensor:
        result = torch.zeros_like(X[..., 0])
        for i in range(self.dim // 4):
            i_ = i + 1
            part1 = (X[..., 4 * i_ - 4] + 10.0 * X[..., 4 * i_ - 3]) ** 2
            part2 = 5.0 * (X[..., 4 * i_ - 2] - X[..., 4 * i_ - 1]) ** 2
            part3 = (X[..., 4 * i_ - 3] - 2.0 * X[..., 4 * i_ - 2]) ** 4
            part4 = 10.0 * (X[..., 4 * i_ - 4] - X[..., 4 * i_ - 1]) ** 4
            result += part1 + part2 + part3 + part4
        return result


class Rastrigin(SyntheticTestFunction):
    _optimal_value = 0.0

    def __init__(self, dim=2, noise_std: Optional[float]=None, negate: bool=False) ->None:
        self.dim = dim
        self._bounds = [(-5.12, 5.12) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) ->Tensor:
        return 10.0 * self.dim + torch.sum(X ** 2 - 10.0 * torch.cos(2.0 * math.pi * X), dim=-1)


class Rosenbrock(SyntheticTestFunction):
    """Rosenbrock synthetic test function.

    d-dimensional function (usually evaluated on `[-5, 10]^d`):

        f(x) = sum_{i=1}^{d-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)

    f has one minimizer for its global minimum at `z_1 = (1, 1, ..., 1)` with
    `f(z_i) = 0.0`.
    """
    _optimal_value = 0.0

    def __init__(self, dim=2, noise_std: Optional[float]=None, negate: bool=False) ->None:
        self.dim = dim
        self._bounds = [(-5.0, 10.0) for _ in range(self.dim)]
        self._optimizers = [tuple(1.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) ->Tensor:
        return torch.sum(100.0 * (X[(...), 1:] - X[(...), :-1] ** 2) ** 2 + (X[(...), :-1] - 1) ** 2, dim=-1)


class Shekel(SyntheticTestFunction):
    """Shekel synthtetic test function.

    4-dimensional function (usually evaluated on `[0, 10]^4`):

        f(x) = -sum_{i=1}^10 (sum_{j=1}^4 (x_j - A_{ji})^2 + C_i)^{-1}

    f has one minimizer for its global minimum at `z_1 = (4, 4, 4, 4)` with
    `f(z_1) = -10.5363`.
    """
    dim = 4
    _bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    _optimizers = [(4.000747, 3.99951, 4.00075, 3.99951)]

    def __init__(self, m: int=10, noise_std: Optional[float]=None, negate: bool=False) ->None:
        self.m = m
        optvals = {(5): -10.1532, (7): -10.4029, (10): -10.536443}
        self._optimal_value = optvals[self.m]
        super().__init__(noise_std=noise_std, negate=negate)
        self.register_buffer('beta', torch.tensor([1, 2, 2, 4, 4, 6, 3, 7, 5, 5], dtype=torch.float))
        C_t = torch.tensor([[4, 1, 8, 6, 3, 2, 5, 8, 6, 7], [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6], [4, 1, 8, 6, 3, 2, 5, 8, 6, 7], [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]], dtype=torch.float)
        self.register_buffer('C', C_t.transpose(-1, -2))

    def evaluate_true(self, X: Tensor) ->Tensor:
        self
        beta = self.beta / 10.0
        result = -sum(1 / (torch.sum((X - self.C[i]) ** 2, dim=-1) + beta[i]) for i in range(self.m))
        return result


class SixHumpCamel(SyntheticTestFunction):
    dim = 2
    _bounds = [(-3.0, 3.0), (-2.0, 2.0)]
    _optimal_value = -1.0316
    _optimizers = [(0.0898, -0.7126), (-0.0898, 0.7126)]

    def evaluate_true(self, X: Tensor) ->Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        return (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (4 * x2 ** 2 - 4) * x2 ** 2


class StyblinskiTang(SyntheticTestFunction):
    """Styblinski-Tang synthtetic test function.

    d-dimensional function (usually evaluated on the hypercube `[-5, 5]^d`):

        H(x) = 0.5 * sum_{i=1}^d (x_i^4 - 16 * x_i^2 + 5 * x_i)

    H has a single global mininimum `H(z) = -39.166166 * d` at `z = [-2.903534]^d`
    """

    def __init__(self, dim=2, noise_std: Optional[float]=None, negate: bool=False) ->None:
        self.dim = dim
        self._bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        self._optimal_value = -39.166166 * self.dim
        self._optimizers = [tuple(-2.903534 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) ->Tensor:
        return 0.5 * (X ** 4 - 16 * X ** 2 + 5 * X).sum(dim=1)


class ThreeHumpCamel(SyntheticTestFunction):
    dim = 2
    _bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    _optimal_value = 0.0
    _optimizers = [(0.0, 0.0)]

    def evaluate_true(self, X: Tensor) ->Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        return 2.0 * x1 ** 2 - 1.05 * x1 ** 4 + x1 ** 6 / 6.0 + x1 * x2 + x2 ** 2


class MockPosterior(Posterior):
    """Mock object that implements dummy methods and feeds through specified outputs"""

    def __init__(self, mean=None, variance=None, samples=None):
        self._mean = mean
        self._variance = variance
        self._samples = samples

    @property
    def device(self) ->torch.device:
        for t in (self._mean, self._variance, self._samples):
            if torch.is_tensor(t):
                return t.device
        return torch.device('cpu')

    @property
    def dtype(self) ->torch.dtype:
        for t in (self._mean, self._variance, self._samples):
            if torch.is_tensor(t):
                return t.dtype
        return torch.float32

    @property
    def event_shape(self) ->torch.Size:
        if self._samples is not None:
            return self._samples.shape
        if self._mean is not None:
            return self._mean.shape
        if self._variance is not None:
            return self._variance.shape
        return torch.Size()

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    def rsample(self, sample_shape: Optional[torch.Size]=None, base_samples: Optional[Tensor]=None) ->Tensor:
        """Mock sample by repeating self._samples. If base_samples is provided,
        do a shape check but return the same mock samples."""
        if sample_shape is None:
            sample_shape = torch.Size()
        if sample_shape is not None and base_samples is not None:
            if base_samples.shape[:len(sample_shape)] != sample_shape:
                raise RuntimeError('sample_shape disagrees with base_samples.')
        return self._samples.expand(sample_shape + self._samples.shape)


class DummyMCObjective(MCAcquisitionObjective):

    def forward(self, samples: Tensor) ->Tensor:
        return samples.sum(-1)


class NotSoAbstractInputTransform(InputTransform):

    def forward(self, X):
        pass


class NotSoAbstractOutcomeTransform(OutcomeTransform):

    def forward(self, Y, Yvar):
        pass


class MockAcquisitionFunction:
    """Mock acquisition function object that implements dummy methods."""

    def __init__(self):
        self.model = None
        self.X_pending = None

    def __call__(self, X):
        return X[..., 0].max(dim=-1)[0]

    def set_X_pending(self, X_pending: Optional[Tensor]=None):
        self.X_pending = X_pending


class DummyTestProblem(BaseTestProblem):
    dim = 2
    _bounds = [(0, 1), (2, 3)]

    def evaluate_true(self, X: Tensor) ->Tensor:
        return -X.pow(2).sum(dim=-1)


class DummyConstrainedTestProblem(DummyTestProblem, ConstrainedBaseTestProblem):
    num_constraints = 1

    def evaluate_slack_true(self, X: Tensor) ->Tensor:
        return 0.25 - X.sum(dim=-1, keepdim=True)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Ackley,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Beale,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Branin,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Bukin,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChainedInputTransform,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChainedOutcomeTransform,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Cosine8,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DropWave,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DummyConstrainedTestProblem,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DummyMCObjective,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DummyTestProblem,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EggHolder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GenericMCObjective,
     lambda: ([], {'objective': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HolderTable,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (IdentityMCObjective,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Log,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Normalize,
     lambda: ([], {'d': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NotSoAbstractInputTransform,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NotSoAbstractOutcomeTransform,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Shekel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SixHumpCamel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Standardize,
     lambda: ([], {'m': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ThreeHumpCamel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_pytorch_botorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

