import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
stable_baselines3 = _module
a2c = _module
a2c = _module
common = _module
atari_wrappers = _module
base_class = _module
bit_flipping_env = _module
buffers = _module
callbacks = _module
cmd_util = _module
distributions = _module
env_checker = _module
evaluation = _module
identity_env = _module
logger = _module
monitor = _module
noise = _module
policies = _module
preprocessing = _module
results_plotter = _module
running_mean_std = _module
save_util = _module
type_aliases = _module
utils = _module
vec_env = _module
base_vec_env = _module
dummy_vec_env = _module
subproc_vec_env = _module
util = _module
vec_check_nan = _module
vec_frame_stack = _module
vec_normalize = _module
vec_transpose = _module
vec_video_recorder = _module
ppo = _module
policies = _module
ppo = _module
sac = _module
policies = _module
sac = _module
td3 = _module
policies = _module
td3 = _module
tests = _module
test_callbacks = _module
test_cnn = _module
test_custom_policy = _module
test_deterministic = _module
test_distributions = _module
test_envs = _module
test_identity = _module
test_logger = _module
test_monitor = _module
test_predict = _module
test_run = _module
test_save_load = _module
test_sde = _module
test_spaces = _module
test_utils = _module
test_vec_check_nan = _module
test_vec_envs = _module
test_vec_normalize = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch as th


import torch.nn.functional as F


from typing import Type


from typing import Union


from typing import Callable


from typing import Optional


from typing import Dict


from typing import Any


import time


from typing import List


from typing import Tuple


from abc import ABC


from abc import abstractmethod


from collections import deque


import numpy as np


from typing import Generator


import torch.nn as nn


from torch.distributions import Normal


from torch.distributions import Categorical


from torch.distributions import Bernoulli


from itertools import zip_longest


from typing import NamedTuple


import random


from functools import partial


from copy import deepcopy


def get_device(device: Union[th.device, str]='auto') ->th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: (Union[str, th.device]) One for 'auto', 'cuda', 'cpu'
    :return: (th.device)
    """
    if device == 'auto':
        device = 'cuda'
    device = th.device(device)
    if device == th.device('cuda') and not th.cuda.is_available():
        return th.device('cpu')
    return device


class MlpExtractor(nn.Module):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
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

    :param feature_dim: (int) Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: (Type[nn.Module]) The activation function to use for the networks.
    :param device: (th.device)
    """

    def __init__(self, feature_dim: int, net_arch: List[Union[int, Dict[str, List[int]]]], activation_fn: Type[nn.Module], device: Union[th.device, str]='auto'):
        super(MlpExtractor, self).__init__()
        device = get_device(device)
        shared_net, policy_net, value_net = [], [], []
        policy_only_layers = []
        value_only_layers = []
        last_layer_dim_shared = feature_dim
        for idx, layer in enumerate(net_arch):
            if isinstance(layer, int):
                layer_size = layer
                shared_net.append(nn.Linear(last_layer_dim_shared, layer_size))
                shared_net.append(activation_fn())
                last_layer_dim_shared = layer_size
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
        for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
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
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)


class Distribution(object):

    def __init__(self):
        super(Distribution, self).__init__()

    def log_prob(self, x: th.Tensor) ->th.Tensor:
        """
        returns the log likelihood

        :param x: (th.Tensor) the taken action
        :return: (th.Tensor) The log likelihood of the distribution
        """
        raise NotImplementedError

    def entropy(self) ->Optional[th.Tensor]:
        """
        Returns Shannon's entropy of the probability

        :return: (Optional[th.Tensor]) the entropy,
            return None if no analytical form is known
        """
        raise NotImplementedError

    def sample(self) ->th.Tensor:
        """
        Returns a sample from the probability distribution

        :return: (th.Tensor) the stochastic action
        """
        raise NotImplementedError

    def mode(self) ->th.Tensor:
        """
        Returns the most likely action (deterministic output)
        from the probability distribution

        :return: (th.Tensor) the stochastic action
        """
        raise NotImplementedError

    def get_actions(self, deterministic: bool=False) ->th.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic: (bool)
        :return: (th.Tensor)
        """
        if deterministic:
            return self.mode()
        return self.sample()

    def actions_from_params(self, *args, **kwargs) ->th.Tensor:
        """
        Returns samples from the probability distribution
        given its parameters.

        :return: (th.Tensor) actions
        """
        raise NotImplementedError

    def log_prob_from_params(self, *args, **kwargs) ->Tuple[th.Tensor, th.Tensor]:
        """
        Returns samples and the associated log probabilities
        from the probability distribution given its parameters.

        :return: (th.Tuple[th.Tensor, th.Tensor]) actions and log prob
        """
        raise NotImplementedError


class BernoulliDistribution(Distribution):
    """
    Bernoulli distribution for MultiBinary action spaces.

    :param action_dim: (int) Number of binary actions
    """

    def __init__(self, action_dims: int):
        super(BernoulliDistribution, self).__init__()
        self.distribution = None
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) ->nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Bernoulli distribution.

        :param latent_dim: (int) Dimension of the last layer
            of the policy network (before the action layer)
        :return: (nn.Linear)
        """
        action_logits = nn.Linear(latent_dim, self.action_dims)
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) ->'BernoulliDistribution':
        self.distribution = Bernoulli(logits=action_logits)
        return self

    def mode(self) ->th.Tensor:
        return th.round(self.distribution.probs)

    def sample(self) ->th.Tensor:
        return self.distribution.sample()

    def entropy(self) ->th.Tensor:
        return self.distribution.entropy().sum(dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool=False) ->th.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) ->Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def log_prob(self, actions: th.Tensor) ->th.Tensor:
        return self.distribution.log_prob(actions).sum(dim=1)


class CategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete actions.

    :param action_dim: (int) Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super(CategoricalDistribution, self).__init__()
        self.distribution = None
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) ->nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: (int) Dimension of the last layer
            of the policy network (before the action layer)
        :return: (nn.Linear)
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) ->'CategoricalDistribution':
        self.distribution = Categorical(logits=action_logits)
        return self

    def mode(self) ->th.Tensor:
        return th.argmax(self.distribution.probs, dim=1)

    def sample(self) ->th.Tensor:
        return self.distribution.sample()

    def entropy(self) ->th.Tensor:
        return self.distribution.entropy()

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool=False) ->th.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) ->Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def log_prob(self, actions: th.Tensor) ->th.Tensor:
        return self.distribution.log_prob(actions)


def sum_independent_dims(tensor: th.Tensor) ->th.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum the components for the ``log_prob``
    or the entropy.

    :param tensor: (th.Tensor) shape: (n_batch, n_actions) or (n_batch,)
    :return: (th.Tensor) shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class DiagGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix,
    for continuous actions.

    :param action_dim: (int)  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super(DiagGaussianDistribution, self).__init__()
        self.distribution = None
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float=0.0) ->Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: (int) Dimension og the last layer of the policy (before the action layer)
        :param log_std_init: (float) Initial value for the log standard deviation
        :return: (nn.Linear, nn.Parameter)
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor) ->'DiagGaussianDistribution':
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions: (th.Tensor)
        :param log_std: (th.Tensor)
        :return: (DiagGaussianDistribution)
        """
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def mode(self) ->th.Tensor:
        return self.distribution.mean

    def sample(self) ->th.Tensor:
        return self.distribution.rsample()

    def entropy(self) ->th.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, deterministic: bool=False) ->th.Tensor:
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) ->Tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions: (th.Tensor)
        :param log_std: (th.Tensor)
        :return: (Tuple[th.Tensor, th.Tensor])
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def log_prob(self, actions: th.Tensor) ->th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must call ``proba_distribution()`` method before.

        :param actions: (th.Tensor)
        :return: (th.Tensor)
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)


class MultiCategoricalDistribution(Distribution):
    """
    MultiCategorical distribution for multi discrete actions.

    :param action_dims: (List[int]) List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: List[int]):
        super(MultiCategoricalDistribution, self).__init__()
        self.action_dims = action_dims
        self.distributions = None

    def proba_distribution_net(self, latent_dim: int) ->nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        :param latent_dim: (int) Dimension of the last layer
            of the policy network (before the action layer)
        :return: (nn.Linear)
        """
        action_logits = nn.Linear(latent_dim, sum(self.action_dims))
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) ->'MultiCategoricalDistribution':
        self.distributions = [Categorical(logits=split) for split in th.split(action_logits, tuple(self.action_dims), dim=1)]
        return self

    def mode(self) ->th.Tensor:
        return th.stack([th.argmax(dist.probs, dim=1) for dist in self.distributions], dim=1)

    def sample(self) ->th.Tensor:
        return th.stack([dist.sample() for dist in self.distributions], dim=1)

    def entropy(self) ->th.Tensor:
        return th.stack([dist.entropy() for dist in self.distributions], dim=1).sum(dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool=False) ->th.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) ->Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def log_prob(self, actions: th.Tensor) ->th.Tensor:
        return th.stack([dist.log_prob(action) for dist, action in zip(self.distributions, th.unbind(actions, dim=1))], dim=1).sum(dim=1)


class TanhBijector(object):
    """
    Bijective transformation of a probability distribution
    using a squashing function (tanh)
    TODO: use Pyro instead (https://pyro.ai/)

    :param epsilon: (float) small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, epsilon: float=1e-06):
        super(TanhBijector, self).__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: th.Tensor) ->th.Tensor:
        return th.tanh(x)

    @staticmethod
    def atanh(x: th.Tensor) ->th.Tensor:
        """
        Inverse of Tanh

        Taken from pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: th.Tensor) ->th.Tensor:
        """
        Inverse tanh.

        :param y: (th.Tensor)
        :return: (th.Tensor)
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

    :param action_dim: (int) Dimension of the action space.
    :param full_std: (bool) Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,)
    :param use_expln: (bool) Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: (bool) Whether to squash the output using a tanh function,
        this allows to ensure boundaries.
    :param learn_features: (bool) Whether to learn features for gSDE or not.
        This will enable gradients to be backpropagated through the features
        ``latent_sde`` in the code.
    :param epsilon: (float) small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, action_dim: int, full_std: bool=True, use_expln: bool=False, squash_output: bool=False, learn_features: bool=False, epsilon: float=1e-06):
        super(StateDependentNoiseDistribution, self).__init__()
        self.distribution = None
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

        :param log_std: (th.Tensor)
        :return: (th.Tensor)
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

        :param log_std: (th.Tensor)
        :param batch_size: (int)
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

        :param latent_dim: (int) Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: (float) Initial value for the log standard deviation
        :param latent_sde_dim: (Optional[int]) Dimension of the last layer of the feature extractor
            for gSDE. By default, it is shared with the policy network.
        :return: (nn.Linear, nn.Parameter)
        """
        mean_actions_net = nn.Linear(latent_dim, self.action_dim)
        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim
        log_std = th.ones(self.latent_sde_dim, self.action_dim) if self.full_std else th.ones(self.latent_sde_dim, 1)
        log_std = nn.Parameter(log_std * log_std_init, requires_grad=True)
        self.sample_weights(log_std)
        return mean_actions_net, log_std

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor) ->'StateDependentNoiseDistribution':
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions: (th.Tensor)
        :param log_std: (th.Tensor)
        :param latent_sde: (th.Tensor)
        :return: (StateDependentNoiseDistribution)
        """
        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        variance = th.mm(latent_sde ** 2, self.get_std(log_std) ** 2)
        self.distribution = Normal(mean_actions, th.sqrt(variance + self.epsilon))
        return self

    def mode(self) ->th.Tensor:
        actions = self.distribution.mean
        if self.bijector is not None:
            return self.bijector.forward(actions)
        return actions

    def get_noise(self, latent_sde: th.Tensor) ->th.Tensor:
        latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return th.mm(latent_sde, self.exploration_mat)
        latent_sde = latent_sde.unsqueeze(1)
        noise = th.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(1)

    def sample(self) ->th.Tensor:
        noise = self.get_noise(self._latent_sde)
        actions = self.distribution.mean + noise
        if self.bijector is not None:
            return self.bijector.forward(actions)
        return actions

    def entropy(self) ->Optional[th.Tensor]:
        if self.bijector is not None:
            return None
        return sum_independent_dims(self.distribution.entropy())

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor, deterministic: bool=False) ->th.Tensor:
        self.proba_distribution(mean_actions, log_std, latent_sde)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, latent_sde: th.Tensor) ->Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(mean_actions, log_std, latent_sde)
        log_prob = self.log_prob(actions)
        return actions, log_prob

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


def create_mlp(input_dim: int, output_dim: int, net_arch: List[int], activation_fn: Type[nn.Module]=nn.ReLU, squash_output: bool=False) ->List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: (int) Dimension of the input vector
    :param output_dim: (int)
    :param net_arch: (List[int]) Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: (Type[nn.Module]) The activation function
        to use after each layer.
    :param squash_output: (bool) Whether to squash the output using a Tanh
        activation function
    :return: (List[nn.Module])
    """
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []
    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())
    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


def create_sde_features_extractor(features_dim: int, sde_net_arch: List[int], activation_fn: Type[nn.Module]) ->Tuple[nn.Sequential, int]:
    """
    Create the neural network that will be used to extract features
    for the gSDE exploration function.

    :param features_dim: (int)
    :param sde_net_arch: ([int])
    :param activation_fn: (Type[nn.Module])
    :return: (nn.Sequential, int)
    """
    sde_activation = activation_fn if len(sde_net_arch) > 0 else None
    latent_sde_net = create_mlp(features_dim, -1, sde_net_arch, activation_fn=sde_activation, squash_output=False)
    latent_sde_dim = sde_net_arch[-1] if len(sde_net_arch) > 0 else features_dim
    sde_features_extractor = nn.Sequential(*latent_sde_net)
    return sde_features_extractor, latent_sde_dim


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

