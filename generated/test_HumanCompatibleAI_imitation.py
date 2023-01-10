import sys
_module = sys.modules[__name__]
del sys
util = _module
check_typeignore = _module
clean_notebooks = _module
conf = _module
quickstart = _module
convert_traj = _module
setup = _module
imitation = _module
algorithms = _module
adversarial = _module
airl = _module
common = _module
gail = _module
base = _module
bc = _module
dagger = _module
density = _module
mce_irl = _module
preference_comparisons = _module
data = _module
buffer = _module
rollout = _module
types = _module
wrappers = _module
policies = _module
base = _module
exploration_wrapper = _module
replay_buffer_wrapper = _module
serialize = _module
regularization = _module
regularizers = _module
updaters = _module
rewards = _module
reward_function = _module
reward_nets = _module
reward_wrapper = _module
serialize = _module
scripts = _module
analyze = _module
config = _module
eval_policy = _module
parallel = _module
train_adversarial = _module
train_imitation = _module
train_preference_comparisons = _module
train_rl = _module
convert_trajs = _module
ingredients = _module
demonstrations = _module
environment = _module
expert = _module
logging = _module
reward = _module
rl = _module
train = _module
wb = _module
train_adversarial = _module
train_preference_comparisons = _module
testing = _module
expert_trajectories = _module
reward_improvement = _module
reward_nets = _module
logger = _module
networks = _module
registry = _module
sacred = _module
util = _module
video_wrapper = _module
conftest = _module
test_adversarial = _module
test_base = _module
test_bc = _module
test_dagger = _module
test_density_baselines = _module
test_mce_irl = _module
test_preference_comparisons = _module
conftest = _module
test_buffer = _module
test_rollout = _module
test_types = _module
test_wrappers = _module
test_exploration_wrapper = _module
test_policies = _module
test_replay_buffer_wrapper = _module
test_reward_fn = _module
test_reward_nets = _module
test_reward_wrapper = _module
test_rewards = _module
test_scripts = _module
test_examples = _module
test_experiments = _module
test_regularization = _module
test_logger = _module
test_networks = _module
test_registry = _module
test_util = _module
test_wb_logger = _module

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


from typing import TYPE_CHECKING


from typing import Optional


import torch as th


import abc


import logging


from typing import Callable


from typing import Iterable


from typing import Iterator


from typing import Mapping


from typing import Type


from typing import overload


import numpy as np


import torch.utils.tensorboard as thboard


from torch.nn import functional as F


from typing import Any


from typing import Generic


from typing import TypeVar


from typing import Union


from typing import cast


import torch.utils.data as th_data


import itertools


from typing import Tuple


from typing import List


from typing import Sequence


from torch.utils import data as th_data


import collections


from typing import NoReturn


import scipy.special


import math


import re


from collections import defaultdict


from typing import Dict


from typing import NamedTuple


from scipy import special


from torch import nn


from torch.utils import data as data_th


from typing import Protocol


from torch import optim


import functools


import uuid


import torch.random


import torch


from collections import Counter


import pandas as pd


T = TypeVar('T')


Pair = Tuple[T, T]


def _rews_validation(rews: np.ndarray, acts: np.ndarray):
    if rews.shape != (len(acts),):
        raise ValueError(f'rewards must be 1D array, one entry for each action: {rews.shape} != ({len(acts)},)')
    if not np.issubdtype(rews.dtype, np.floating):
        raise ValueError(f'rewards dtype {rews.dtype} not a float')


TransitionsMinimalSelf = TypeVar('TransitionsMinimalSelf', bound='TransitionsMinimal')


def dataclass_quick_asdict(obj) ->Dict[str, Any]:
    """Extract dataclass to items using `dataclasses.fields` + dict comprehension.

    This is a quick alternative to `dataclasses.asdict`, which expensively and
    undocumentedly deep-copies every numpy array value.
    See https://stackoverflow.com/a/52229565/1091722.

    Args:
        obj: A dataclass instance.

    Returns:
        A dictionary mapping from `obj` field names to values.
    """
    d = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    return d


PolicyCallable = Callable[[np.ndarray], np.ndarray]


class LossAndMetrics(NamedTuple):
    """Loss and auxiliary metrics for reward network training."""
    loss: th.Tensor
    metrics: Mapping[str, th.Tensor]


def cnn_transpose(tens: th.Tensor) ->th.Tensor:
    """Transpose a (b,h,w,c)-formatted tensor to (b,c,h,w) format."""
    if len(tens.shape) == 4:
        return th.permute(tens, (0, 3, 1, 2))
    else:
        raise ValueError(f'Invalid input: len(tens.shape) = {len(tens.shape)} != 4.')


class SqueezeLayer(nn.Module):
    """Torch module that squeezes a B*1 tensor down into a size-B vector."""

    def forward(self, x):
        assert x.ndim == 2 and x.shape[1] == 1
        new_value = x.squeeze(1)
        assert new_value.ndim == 1
        return new_value


class BaseNorm(nn.Module, abc.ABC):
    """Base class for layers that try to normalize the input to mean 0 and variance 1.

    Similar to BatchNorm, LayerNorm, etc. but whereas they only use statistics from
    the current batch at train time, we use statistics from all batches.
    """
    running_mean: th.Tensor
    running_var: th.Tensor
    count: th.Tensor

    def __init__(self, num_features: int, eps: float=1e-05):
        """Builds RunningNorm.

        Args:
            num_features: Number of features; the length of the non-batch dimension.
            eps: Small constant for numerical stability. Inputs are rescaled by
                `1 / sqrt(estimated_variance + eps)`.
        """
        super().__init__()
        self.eps = eps
        self.register_buffer('running_mean', th.empty(num_features))
        self.register_buffer('running_var', th.empty(num_features))
        self.register_buffer('count', th.empty((), dtype=th.int))
        BaseNorm.reset_running_stats(self)

    def reset_running_stats(self) ->None:
        """Resets running stats to defaults, yielding the identity transformation."""
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.count.zero_()

    def forward(self, x: th.Tensor) ->th.Tensor:
        """Updates statistics if in training mode. Returns normalized `x`."""
        if self.training:
            with th.no_grad():
                self.update_stats(x)
        return (x - self.running_mean) / th.sqrt(self.running_var + self.eps)

    @abc.abstractmethod
    def update_stats(self, batch: th.Tensor) ->None:
        """Update `self.running_mean`, `self.running_var` and `self.count`."""


class RunningNorm(BaseNorm):
    """Normalizes input to mean 0 and standard deviation 1 using a running average.

    Similar to BatchNorm, LayerNorm, etc. but whereas they only use statistics from
    the current batch at train time, we use statistics from all batches.

    This should replicate the common practice in RL of normalizing environment
    observations, such as using ``VecNormalize`` in Stable Baselines. Note that
    the behavior of this class is slightly different from `VecNormalize`, e.g.,
    it works with the current reward instead of return estimate, and subtracts the mean
    reward whereas ``VecNormalize`` only rescales it.
    """

    def update_stats(self, batch: th.Tensor) ->None:
        """Update `self.running_mean`, `self.running_var` and `self.count`.

        Uses Chan et al (1979), "Updating Formulae and a Pairwise Algorithm for
        Computing Sample Variances." to update the running moments in a numerically
        stable fashion.

        Args:
            batch: A batch of data to use to update the running mean and variance.
        """
        batch_mean = th.mean(batch, dim=0)
        batch_var = th.var(batch, dim=0, unbiased=False)
        batch_count = batch.shape[0]
        delta = batch_mean - self.running_mean
        tot_count = self.count + batch_count
        self.running_mean += delta * batch_count / tot_count
        self.running_var *= self.count
        self.running_var += batch_var * batch_count
        self.running_var += th.square(delta) * self.count * batch_count / tot_count
        self.running_var /= tot_count
        self.count += batch_count


class EMANorm(BaseNorm):
    """Similar to RunningNorm but uses an exponential weighting."""
    inv_learning_rate: th.Tensor
    num_batches: th.IntTensor

    def __init__(self, num_features: int, decay: float=0.99, eps: float=1e-05):
        """Builds EMARunningNorm.

        Args:
            num_features: Number of features; the length of the non-batch dim.
            decay: how quickly the weight on past samples decays over time.
            eps: small constant for numerical stability.

        Raises:
            ValueError: if decay is out of range.
        """
        super().__init__(num_features, eps=eps)
        if not 0 < decay < 1:
            raise ValueError('decay must be between 0 and 1')
        self.decay = decay
        self.register_buffer('inv_learning_rate', th.empty(()))
        self.register_buffer('num_batches', th.empty((), dtype=th.int))
        EMANorm.reset_running_stats(self)

    def reset_running_stats(self):
        """Reset the running stats of the normalization layer."""
        super().reset_running_stats()
        self.inv_learning_rate.zero_()
        self.num_batches.zero_()

    def update_stats(self, batch: th.Tensor) ->None:
        """Update `self.running_mean` and `self.running_var` in batch mode.

        Reference Algorithm 3 from:
        https://github.com/HumanCompatibleAI/imitation/files/9456540/Incremental_batch_EMA_and_EMV.pdf

        Args:
            batch: A batch of data to use to update the running mean and variance.
        """
        b_size = batch.shape[0]
        if len(batch.shape) == 1:
            batch = batch.reshape(b_size, 1)
        self.inv_learning_rate += self.decay ** self.num_batches
        learning_rate = 1 / self.inv_learning_rate
        delta_mean = batch.mean(0) - self.running_mean
        self.running_mean += learning_rate * delta_mean
        batch_var = batch.var(0, unbiased=False)
        delta_var = batch_var + (1 - learning_rate) * delta_mean ** 2 - self.running_var
        self.running_var += learning_rate * delta_var
        self.count += b_size
        self.num_batches += 1


class ZeroModule(nn.Module):
    """Module that always returns zeros of same shape as input."""

    def __init__(self, features_dim: int):
        """Builds ZeroModule."""
        super().__init__()
        self.features_dim = features_dim

    def forward(self, x: th.Tensor) ->th.Tensor:
        """Returns zeros of same shape as `x`."""
        assert x.shape[1:] == (self.features_dim,)
        return th.zeros_like(x)


class EMANormAlgorithm2(networks.EMANorm):
    """EMA Norm using algorithm 2 from the reference below.

    Reference:
    https://github.com/HumanCompatibleAI/imitation/files/9456540/Incremental_batch_EMA_and_EMV.pdf
    """

    def __init__(self, num_features: int, decay: float=0.99, eps: float=1e-05):
        """Builds EMARunningNormIncremental."""
        super().__init__(num_features, decay=decay, eps=eps)

    def update_stats(self, batch: th.Tensor) ->None:
        """Update `self.running_mean` and `self.running_var` incrementally.

        Reference Algorithm 2 from:
        https://github.com/HumanCompatibleAI/imitation/files/9364938/Incremental_batch_EMA_and_EMV.pdf

        Args:
            batch: A batch of data to use to update the running mean and variance.
        """
        b_size = batch.shape[0]
        if len(batch.shape) == 1:
            batch = batch.reshape(b_size, 1)
        self.inv_learning_rate += self.decay ** self.num_batches
        learning_rate = 1 / self.inv_learning_rate
        if self.count == 0:
            self.running_mean = batch.mean(dim=0)
            if b_size > 1:
                self.running_var = batch.var(dim=0, unbiased=False)
            else:
                self.running_var = th.zeros_like(self.running_mean, dtype=th.float)
        else:
            S = th.mean((batch - self.running_mean) ** 2, dim=0)
            delta = learning_rate * (batch.mean(0) - self.running_mean)
            self.running_mean += delta
            self.running_var *= 1 - learning_rate
            self.running_var += learning_rate * S - delta ** 2
        self.count += b_size
        self.num_batches += 1


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (EMANorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EMANormAlgorithm2,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RunningNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ZeroModule,
     lambda: ([], {'features_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_HumanCompatibleAI_imitation(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

