import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
stable_baselines3 = _module
a2c = _module
a2c = _module
policies = _module
common = _module
atari_wrappers = _module
base_class = _module
buffers = _module
callbacks = _module
distributions = _module
env_checker = _module
env_util = _module
envs = _module
bit_flipping_env = _module
identity_env = _module
multi_input_envs = _module
evaluation = _module
logger = _module
monitor = _module
noise = _module
off_policy_algorithm = _module
on_policy_algorithm = _module
policies = _module
preprocessing = _module
results_plotter = _module
running_mean_std = _module
save_util = _module
sb2_compat = _module
rmsprop_tf_like = _module
torch_layers = _module
type_aliases = _module
utils = _module
vec_env = _module
base_vec_env = _module
dummy_vec_env = _module
stacked_observations = _module
subproc_vec_env = _module
util = _module
vec_check_nan = _module
vec_extract_dict_obs = _module
vec_frame_stack = _module
vec_monitor = _module
vec_normalize = _module
vec_transpose = _module
vec_video_recorder = _module
ddpg = _module
ddpg = _module
dqn = _module
dqn = _module
policies = _module
her = _module
goal_selection_strategy = _module
her_replay_buffer = _module
ppo = _module
ppo = _module
sac = _module
policies = _module
sac = _module
td3 = _module
policies = _module
td3 = _module
tests = _module
test_buffers = _module
test_callbacks = _module
test_cnn = _module
test_custom_policy = _module
test_deterministic = _module
test_dict_env = _module
test_distributions = _module
test_env_checker = _module
test_envs = _module
test_gae = _module
test_her = _module
test_identity = _module
test_logger = _module
test_monitor = _module
test_predict = _module
test_preprocessing = _module
test_run = _module
test_save_load = _module
test_sde = _module
test_spaces = _module
test_tensorboard = _module
test_train_eval_mode = _module
test_utils = _module
test_vec_check_nan = _module
test_vec_envs = _module
test_vec_extract_dict_obs = _module
test_vec_monitor = _module
test_vec_normalize = _module

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


from typing import Dict


from typing import List


from typing import Any


from typing import Optional


from typing import Type


from typing import TypeVar


from typing import Union


import torch as th


from torch.nn import functional as F


import time


import warnings


from abc import ABC


from abc import abstractmethod


from collections import deque


from typing import Iterable


from typing import Tuple


import numpy as np


from typing import Generator


from torch import nn


from torch.distributions import Bernoulli


from torch.distributions import Categorical


from torch.distributions import Normal


from collections import defaultdict


from typing import Sequence


from typing import TextIO


import pandas


from matplotlib import pyplot as plt


from copy import deepcopy


import collections


import copy


from functools import partial


import functools


from typing import Callable


import torch


from torch.optim import Optimizer


from itertools import zip_longest


from enum import Enum


from typing import NamedTuple


import random


from pandas.errors import EmptyDataError


from collections import OrderedDict


import torch.nn as nn


TensorDict = Dict[Union[str, int], th.Tensor]


SelfBaseModel = TypeVar('SelfBaseModel', bound='BaseModel')


def get_device(device: Union[th.device, str]='auto') ->th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    if device == 'auto':
        device = 'cuda'
    device = th.device(device)
    if device.type == th.device('cuda').type and not th.cuda.is_available():
        return th.device('cpu')
    return device


def obs_as_tensor(obs: Union[np.ndarray, Dict[Union[str, int], np.ndarray]], device: th.device) ->Union[th.Tensor, TensorDict]:
    """
    Moves the observation to the given device.

    :param obs:
    :param device: PyTorch device
    :return: PyTorch tensor of the observation on a desired device.
    """
    if isinstance(obs, np.ndarray):
        return th.as_tensor(obs, device=device)
    elif isinstance(obs, dict):
        return {key: th.as_tensor(_obs, device=device) for key, _obs in obs.items()}
    else:
        raise Exception(f'Unrecognized type of observation {type(obs)}')


SelfDistribution = TypeVar('SelfDistribution', bound='Distribution')


class Distribution(ABC):
    """Abstract base class for distributions."""

    def __init__(self):
        super().__init__()
        self.distribution = None

    @abstractmethod
    def proba_distribution_net(self, *args, **kwargs) ->Union[nn.Module, Tuple[nn.Module, nn.Parameter]]:
        """Create the layers and parameters that represent the distribution.

        Subclasses must define this, but the arguments and return type vary between
        concrete classes."""

    @abstractmethod
    def proba_distribution(self: SelfDistribution, *args, **kwargs) ->SelfDistribution:
        """Set parameters of the distribution.

        :return: self
        """

    @abstractmethod
    def log_prob(self, x: th.Tensor) ->th.Tensor:
        """
        Returns the log likelihood

        :param x: the taken action
        :return: The log likelihood of the distribution
        """

    @abstractmethod
    def entropy(self) ->Optional[th.Tensor]:
        """
        Returns Shannon's entropy of the probability

        :return: the entropy, or None if no analytical form is known
        """

    @abstractmethod
    def sample(self) ->th.Tensor:
        """
        Returns a sample from the probability distribution

        :return: the stochastic action
        """

    @abstractmethod
    def mode(self) ->th.Tensor:
        """
        Returns the most likely action (deterministic output)
        from the probability distribution

        :return: the stochastic action
        """

    def get_actions(self, deterministic: bool=False) ->th.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

    @abstractmethod
    def actions_from_params(self, *args, **kwargs) ->th.Tensor:
        """
        Returns samples from the probability distribution
        given its parameters.

        :return: actions
        """

    @abstractmethod
    def log_prob_from_params(self, *args, **kwargs) ->Tuple[th.Tensor, th.Tensor]:
        """
        Returns samples and the associated log probabilities
        from the probability distribution given its parameters.

        :return: actions and log prob
        """


SelfBernoulliDistribution = TypeVar('SelfBernoulliDistribution', bound='BernoulliDistribution')


class BernoulliDistribution(Distribution):
    """
    Bernoulli distribution for MultiBinary action spaces.

    :param action_dim: Number of binary actions
    """

    def __init__(self, action_dims: int):
        super().__init__()
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) ->nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Bernoulli distribution.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dims)
        return action_logits

    def proba_distribution(self: SelfBernoulliDistribution, action_logits: th.Tensor) ->SelfBernoulliDistribution:
        self.distribution = Bernoulli(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) ->th.Tensor:
        return self.distribution.log_prob(actions).sum(dim=1)

    def entropy(self) ->th.Tensor:
        return self.distribution.entropy().sum(dim=1)

    def sample(self) ->th.Tensor:
        return self.distribution.sample()

    def mode(self) ->th.Tensor:
        return th.round(self.distribution.probs)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool=False) ->th.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) ->Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


SelfCategoricalDistribution = TypeVar('SelfCategoricalDistribution', bound='CategoricalDistribution')


class CategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete actions.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) ->nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self: SelfCategoricalDistribution, action_logits: th.Tensor) ->SelfCategoricalDistribution:
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) ->th.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) ->th.Tensor:
        return self.distribution.entropy()

    def sample(self) ->th.Tensor:
        return self.distribution.sample()

    def mode(self) ->th.Tensor:
        return th.argmax(self.distribution.probs, dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool=False) ->th.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) ->Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


SelfDiagGaussianDistribution = TypeVar('SelfDiagGaussianDistribution', bound='DiagGaussianDistribution')


def sum_independent_dims(tensor: th.Tensor) ->th.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class DiagGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float=0.0) ->Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(self: SelfDiagGaussianDistribution, mean_actions: th.Tensor, log_std: th.Tensor) ->SelfDiagGaussianDistribution:
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: th.Tensor) ->th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) ->th.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) ->th.Tensor:
        return self.distribution.rsample()

    def mode(self) ->th.Tensor:
        return self.distribution.mean

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, deterministic: bool=False) ->th.Tensor:
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) ->Tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class MlpExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.
    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    Adapted from Stable Baselines.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device:
    """

    def __init__(self, feature_dim: int, net_arch: List[Union[int, Dict[str, List[int]]]], activation_fn: Type[nn.Module], device: Union[th.device, str]='auto') ->None:
        super().__init__()
        device = get_device(device)
        shared_net: List[nn.Module] = []
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        policy_only_layers: List[int] = []
        value_only_layers: List[int] = []
        last_layer_dim_shared = feature_dim
        for layer in net_arch:
            if isinstance(layer, int):
                shared_net.append(nn.Linear(last_layer_dim_shared, layer))
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer
            else:
                assert isinstance(layer, dict), 'Error: the net_arch list can only contain ints and dicts'
                if 'pi' in layer:
                    assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer['pi']
                if 'vf' in layer:
                    assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer['vf']
                break
        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared
        for pi_layer_size, vf_layer_size in zip_longest(policy_only_layers, value_only_layers):
            if pi_layer_size is not None:
                assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size))
                policy_net.append(activation_fn())
                last_layer_dim_pi = pi_layer_size
            if vf_layer_size is not None:
                assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_vf, vf_layer_size))
                value_net.append(activation_fn())
                last_layer_dim_vf = vf_layer_size
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.shared_net = nn.Sequential(*shared_net)
        self.policy_net = nn.Sequential(*policy_net)
        self.value_net = nn.Sequential(*value_net)

    def forward(self, features: th.Tensor) ->Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)

    def forward_actor(self, features: th.Tensor) ->th.Tensor:
        return self.policy_net(self.shared_net(features))

    def forward_critic(self, features: th.Tensor) ->th.Tensor:
        return self.value_net(self.shared_net(features))


SelfMultiCategoricalDistribution = TypeVar('SelfMultiCategoricalDistribution', bound='MultiCategoricalDistribution')


class MultiCategoricalDistribution(Distribution):
    """
    MultiCategorical distribution for multi discrete actions.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: List[int]):
        super().__init__()
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) ->nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, sum(self.action_dims))
        return action_logits

    def proba_distribution(self: SelfMultiCategoricalDistribution, action_logits: th.Tensor) ->SelfMultiCategoricalDistribution:
        self.distribution = [Categorical(logits=split) for split in th.split(action_logits, tuple(self.action_dims), dim=1)]
        return self

    def log_prob(self, actions: th.Tensor) ->th.Tensor:
        return th.stack([dist.log_prob(action) for dist, action in zip(self.distribution, th.unbind(actions, dim=1))], dim=1).sum(dim=1)

    def entropy(self) ->th.Tensor:
        return th.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)

    def sample(self) ->th.Tensor:
        return th.stack([dist.sample() for dist in self.distribution], dim=1)

    def mode(self) ->th.Tensor:
        return th.stack([th.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool=False) ->th.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) ->Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


Schedule = Callable[[float], float]


SelfStateDependentNoiseDistribution = TypeVar('SelfStateDependentNoiseDistribution', bound='StateDependentNoiseDistribution')


class TanhBijector:
    """
    Bijective transformation of a probability distribution
    using a squashing function (tanh)
    TODO: use Pyro instead (https://pyro.ai/)

    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, epsilon: float=1e-06):
        super().__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: th.Tensor) ->th.Tensor:
        return th.tanh(x)

    @staticmethod
    def atanh(x: th.Tensor) ->th.Tensor:
        """
        Inverse of Tanh

        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: th.Tensor) ->th.Tensor:
        """
        Inverse tanh.

        :param y:
        :return:
        """
        eps = th.finfo(y.dtype).eps
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: th.Tensor) ->th.Tensor:
        return th.log(1.0 - th.tanh(x) ** 2 + self.epsilon)


class StateDependentNoiseDistribution(Distribution):
    """
    Distribution class for using generalized State Dependent Exploration (gSDE).
    Paper: https://arxiv.org/abs/2005.05719

    It is used to create the noise exploration matrix and
    compute the log probability of an action with that noise.

    :param action_dim: Dimension of the action space.
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,)
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this ensures bounds are satisfied.
    :param learn_features: Whether to learn features for gSDE or not.
        This will enable gradients to be backpropagated through the features
        ``latent_sde`` in the code.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, action_dim: int, full_std: bool=True, use_expln: bool=False, squash_output: bool=False, learn_features: bool=False, epsilon: float=1e-06):
        super().__init__()
        self.action_dim = action_dim
        self.latent_sde_dim = None
        self.mean_actions = None
        self.log_std = None
        self.weights_dist = None
        self.exploration_mat = None
        self.exploration_matrices = None
        self._latent_sde = None
        self.use_expln = use_expln
        self.full_std = full_std
        self.epsilon = epsilon
        self.learn_features = learn_features
        if squash_output:
            self.bijector = TanhBijector(epsilon)
        else:
            self.bijector = None

    def get_std(self, log_std: th.Tensor) ->th.Tensor:
        """
        Get the standard deviation from the learned parameter
        (log of it by default). This ensures that the std is positive.

        :param log_std:
        :return:
        """
        if self.use_expln:
            below_threshold = th.exp(log_std) * (log_std <= 0)
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (th.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            std = th.exp(log_std)
        if self.full_std:
            return std
        return th.ones(self.latent_sde_dim, self.action_dim) * std

    def sample_weights(self, log_std: th.Tensor, batch_size: int=1) ->None:
        """
        Sample weights for the noise exploration matrix,
        using a centered Gaussian distribution.

        :param log_std:
        :param batch_size:
        """
        std = self.get_std(log_std)
        self.weights_dist = Normal(th.zeros_like(std), std)
        self.exploration_mat = self.weights_dist.rsample()
        self.exploration_matrices = self.weights_dist.rsample((batch_size,))

    def proba_distribution_net(self, latent_dim: int, log_std_init: float=-2.0, latent_sde_dim: Optional[int]=None) ->Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the deterministic action, the other parameter will be the
        standard deviation of the distribution that control the weights of the noise matrix.

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :param latent_sde_dim: Dimension of the last layer of the features extractor
            for gSDE. By default, it is shared with the policy network.
        :return:
        """
        mean_actions_net = nn.Linear(latent_dim, self.action_dim)
        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim
        log_std = th.ones(self.latent_sde_dim, self.action_dim) if self.full_std else th.ones(self.latent_sde_dim, 1)
        log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)
        self.sample_weights(log_std)
        return mean_actions_net, log_std

    def proba_distribution(self: SelfStateDependentNoiseDistribution, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor) ->SelfStateDependentNoiseDistribution:
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :param latent_sde:
        :return:
        """
        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        variance = th.mm(self._latent_sde ** 2, self.get_std(log_std) ** 2)
        self.distribution = Normal(mean_actions, th.sqrt(variance + self.epsilon))
        return self

    def log_prob(self, actions: th.Tensor) ->th.Tensor:
        if self.bijector is not None:
            gaussian_actions = self.bijector.inverse(actions)
        else:
            gaussian_actions = actions
        log_prob = self.distribution.log_prob(gaussian_actions)
        log_prob = sum_independent_dims(log_prob)
        if self.bijector is not None:
            log_prob -= th.sum(self.bijector.log_prob_correction(gaussian_actions), dim=1)
        return log_prob

    def entropy(self) ->Optional[th.Tensor]:
        if self.bijector is not None:
            return None
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) ->th.Tensor:
        noise = self.get_noise(self._latent_sde)
        actions = self.distribution.mean + noise
        if self.bijector is not None:
            return self.bijector.forward(actions)
        return actions

    def mode(self) ->th.Tensor:
        actions = self.distribution.mean
        if self.bijector is not None:
            return self.bijector.forward(actions)
        return actions

    def get_noise(self, latent_sde: th.Tensor) ->th.Tensor:
        latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return th.mm(latent_sde, self.exploration_mat)
        latent_sde = latent_sde.unsqueeze(dim=1)
        noise = th.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(dim=1)

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor, deterministic: bool=False) ->th.Tensor:
        self.proba_distribution(mean_actions, log_std, latent_sde)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor) ->Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(mean_actions, log_std, latent_sde)
        log_prob = self.log_prob(actions)
        return actions, log_prob


def create_mlp(input_dim: int, output_dim: int, net_arch: List[int], activation_fn: Type[nn.Module]=nn.ReLU, squash_output: bool=False, with_bias: bool=True) ->List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :return:
    """
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias), activation_fn()]
    else:
        modules = []
    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        modules.append(activation_fn())
    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


def get_actor_critic_arch(net_arch: Union[List[int], Dict[str, List[int]]]) ->Tuple[List[int], List[int]]:
    """
    Get the actor and critic network architectures for off-policy actor-critic algorithms (SAC, TD3, DDPG).

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers,
    which can be different for the actor and the critic.
    It is assumed to be a list of ints or a dict.

    1. If it is a list, actor and critic networks will have the same architecture.
        The architecture is represented by a list of integers (of arbitrary length (zero allowed))
        each specifying the number of units per layer.
       If the number of ints is zero, the network will be linear.
    2. If it is a dict,  it should have the following structure:
       ``dict(qf=[<critic network architecture>], pi=[<actor network architecture>])``.
       where the network architecture is a list as described in 1.

    For example, to have actor and critic that share the same network architecture,
    you only need to specify ``net_arch=[256, 256]`` (here, two hidden layers of 256 units each).

    If you want a different architecture for the actor and the critic,
    then you can specify ``net_arch=dict(qf=[400, 300], pi=[64, 64])``.

    .. note::
        Compared to their on-policy counterparts, no shared layers (other than the features extractor)
        between the actor and the critic are allowed (to prevent issues with target networks).

    :param net_arch: The specification of the actor and critic networks.
        See above for details on its formatting.
    :return: The network architectures for the actor and the critic
    """
    if isinstance(net_arch, list):
        actor_arch, critic_arch = net_arch, net_arch
    else:
        assert isinstance(net_arch, dict), 'Error: the net_arch can only contain be a list of ints or a dict'
        assert 'pi' in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
        assert 'qf' in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
        actor_arch, critic_arch = net_arch['pi'], net_arch['qf']
    return actor_arch, critic_arch


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MlpExtractor,
     lambda: ([], {'feature_dim': 4, 'net_arch': [4, 4], 'activation_fn': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_DLR_RM_stable_baselines3(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

