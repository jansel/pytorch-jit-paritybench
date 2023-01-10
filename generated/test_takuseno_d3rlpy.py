import sys
_module = sys.modules[__name__]
del sys
d3rlpy = _module
_version = _module
algos = _module
awac = _module
base = _module
bc = _module
bcq = _module
bear = _module
combo = _module
cql = _module
crr = _module
ddpg = _module
dqn = _module
iql = _module
mopo = _module
nfq = _module
plas = _module
random_policy = _module
sac = _module
td3 = _module
td3_plus_bc = _module
awac_impl = _module
base = _module
bc_impl = _module
bcq_impl = _module
bear_impl = _module
combo_impl = _module
cql_impl = _module
crr_impl = _module
ddpg_impl = _module
dqn_impl = _module
iql_impl = _module
plas_impl = _module
sac_impl = _module
td3_impl = _module
td3_plus_bc_impl = _module
utility = _module
argument_utility = _module
cli = _module
constants = _module
containers = _module
context = _module
datasets = _module
decorators = _module
dynamics = _module
probabilistic_ensemble_dynamics = _module
base = _module
probabilistic_ensemble_dynamics_impl = _module
envs = _module
wrappers = _module
gpu = _module
iterators = _module
random_iterator = _module
round_iterator = _module
itertools = _module
logger = _module
metrics = _module
comparer = _module
scorer = _module
models = _module
builders = _module
encoders = _module
optimizers = _module
q_functions = _module
distributions = _module
dynamics = _module
encoders = _module
imitators = _module
parameters = _module
policies = _module
q_functions = _module
base = _module
ensemble_q_function = _module
fqf_q_function = _module
iqn_q_function = _module
mean_q_function = _module
qr_q_function = _module
utility = _module
v_functions = _module
online = _module
buffers = _module
explorers = _module
ope = _module
fqe = _module
fqe_impl = _module
preprocessing = _module
action_scalers = _module
reward_scalers = _module
scalers = _module
stack = _module
torch_utility = _module
sb3 = _module
conf = _module
train_bc = _module
train_cql = _module
train_dqn = _module
train_atari = _module
train_discrete_sac = _module
train_sac = _module
fqe_atari = _module
fqe_pybullet = _module
sac_online_dist = _module
sac_online_multistep = _module
td3_plus_bc_dist = _module
td3_plus_bc_multistep = _module
iql = _module
discrete_bcq = _module
discrete_cql = _module
iql = _module
plas_with_perturbation = _module
qr_dqn = _module
double_dqn = _module
fqf = _module
iqn = _module
setup = _module
tests = _module
algo_test = _module
test_awac = _module
test_bc = _module
test_bcq = _module
test_bear = _module
test_combo = _module
test_cql = _module
test_crr = _module
test_ddpg = _module
test_dqn = _module
test_iql = _module
test_mopo = _module
test_nfq = _module
test_plas = _module
test_random_policy = _module
test_sac = _module
test_td3 = _module
test_td3_plus_bc = _module
test_awac_impl = _module
test_bc_impl = _module
test_bcq_impl = _module
test_bear_impl = _module
test_combo_impl = _module
test_cql_impl = _module
test_crr_impl = _module
test_ddpg_impl = _module
test_dqn_impl = _module
test_iql_impl = _module
test_plas_impl = _module
test_sac_impl = _module
test_td3_impl = _module
test_td3_plus_bc_impl = _module
base_test = _module
dummy_env = _module
dynamics_test = _module
test_probabilistic_ensemble_dynamics = _module
test_probabilistic_ensemble_dynamics_impl = _module
test_wrappers = _module
test_random_iterator = _module
test_round_iterator = _module
test_comparer = _module
test_scorer = _module
test_builders = _module
test_encoders = _module
test_optimizers = _module
test_q_functions = _module
model_test = _module
test_ensemble_q_function = _module
test_fqf_q_function = _module
test_iqn_q_function = _module
test_mean_q_function = _module
test_qr_q_function = _module
test_utility = _module
test_distributions = _module
test_dynamics = _module
test_encoders = _module
test_imitators = _module
test_parameters = _module
test_policies = _module
test_q_functions = _module
test_v_functions = _module
test_buffers = _module
test_explorers = _module
test_iterators = _module
test_fqe = _module
test_fqe_impl = _module
test_action_scalers = _module
test_reward_scalers = _module
test_scalers = _module
test_stack = _module
test_argument_utility = _module
test_containers = _module
test_context = _module
test_dataset = _module
test_datasets = _module
test_gpu = _module
test_itertools = _module
test_torch_utility = _module
test_sb3 = _module

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


import random


import numpy as np


import torch


from typing import Any


from typing import Dict


from typing import Optional


from typing import Sequence


from typing import List


from typing import Union


from typing import Tuple


import torch.nn.functional as F


from typing import cast


from torch.optim import Optimizer


from abc import ABCMeta


from abc import abstractmethod


import math


import copy


from torch import nn


from typing import ClassVar


from typing import Type


from typing import Iterable


from torch import optim


from torch.optim import SGD


from torch.optim import Adam


from torch.optim import RMSprop


from torch.distributions import Normal


from torch.nn.utils import spectral_norm


from torch.distributions.kl import kl_divergence


from torch.distributions import Categorical


import collections


from inspect import signature


from typing import Callable


from torch.utils.data._utils.collate import default_collate


from torch.optim.lr_scheduler import CosineAnnealingLR


from sklearn.model_selection import train_test_split


class EncoderWithAction(metaclass=ABCMeta):

    @abstractmethod
    def forward(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        pass

    @abstractmethod
    def get_feature_size(self) ->int:
        pass

    @property
    def action_size(self) ->int:
        pass

    @property
    def observation_shape(self) ->Sequence[int]:
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        pass

    @property
    def last_layer(self) ->nn.Linear:
        raise NotImplementedError


def _apply_spectral_norm_recursively(model: nn.Module) ->None:
    for _, module in model.named_children():
        if isinstance(module, nn.ModuleList):
            for m in module:
                _apply_spectral_norm_recursively(m)
        elif 'weight' in module._parameters:
            spectral_norm(module)


def _gaussian_likelihood(x: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor) ->torch.Tensor:
    inv_var = torch.exp(-2.0 * logstd)
    return 0.5 * ((mu - x) ** 2 * inv_var).sum(dim=1, keepdim=True)


class ProbabilisticDynamicsModel(nn.Module):
    """Probabilistic dynamics model.

    References:
        * `Janner et al., When to Trust Your Model: Model-Based Policy
          Optimization. <https://arxiv.org/abs/1906.08253>`_
        * `Chua et al., Deep Reinforcement Learning in a Handful of Trials
          using Probabilistic Dynamics Models.
          <https://arxiv.org/abs/1805.12114>`_

    """
    _encoder: EncoderWithAction
    _mu: nn.Linear
    _logstd: nn.Linear
    _max_logstd: nn.Parameter
    _min_logstd: nn.Parameter

    def __init__(self, encoder: EncoderWithAction):
        super().__init__()
        _apply_spectral_norm_recursively(cast(nn.Module, encoder))
        self._encoder = encoder
        feature_size = encoder.get_feature_size()
        observation_size = encoder.observation_shape[0]
        out_size = observation_size + 1
        self._mu = spectral_norm(nn.Linear(feature_size, out_size))
        self._logstd = nn.Linear(feature_size, out_size)
        init_max = torch.empty(1, out_size, dtype=torch.float32).fill_(2.0)
        init_min = torch.empty(1, out_size, dtype=torch.float32).fill_(-10.0)
        self._max_logstd = nn.Parameter(init_max)
        self._min_logstd = nn.Parameter(init_min)

    def compute_stats(self, x: torch.Tensor, action: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        h = self._encoder(x, action)
        mu = self._mu(h)
        logstd = self._logstd(h)
        logstd = self._max_logstd - F.softplus(self._max_logstd - logstd)
        logstd = self._min_logstd + F.softplus(logstd - self._min_logstd)
        return mu, logstd

    def forward(self, x: torch.Tensor, action: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        return self.predict_with_variance(x, action)[:2]

    def predict_with_variance(self, x: torch.Tensor, action: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logstd = self.compute_stats(x, action)
        dist = Normal(mu, logstd.exp())
        pred = dist.rsample()
        next_x = x + pred[:, :-1]
        next_reward = pred[:, -1].view(-1, 1)
        return next_x, next_reward, dist.variance.sum(dim=1, keepdims=True)

    def compute_error(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor) ->torch.Tensor:
        mu, logstd = self.compute_stats(observations, actions)
        mu_x = observations + mu[:, :-1]
        mu_reward = mu[:, -1].view(-1, 1)
        logstd_x = logstd[:, :-1]
        logstd_reward = logstd[:, -1].view(-1, 1)
        likelihood_loss = _gaussian_likelihood(next_observations, mu_x, logstd_x)
        likelihood_loss += _gaussian_likelihood(rewards, mu_reward, logstd_reward)
        penalty = logstd.sum(dim=1, keepdim=True)
        bound_loss = self._max_logstd.sum() - self._min_logstd.sum()
        loss = likelihood_loss + penalty + 0.01 * bound_loss
        return loss.view(-1, 1)


def _compute_ensemble_variance(observations: torch.Tensor, rewards: torch.Tensor, variances: torch.Tensor, variance_type: str) ->torch.Tensor:
    if variance_type == 'max':
        return variances.max(dim=1).values
    elif variance_type == 'data':
        data = torch.cat([observations, rewards], dim=2)
        return (data.std(dim=1) ** 2).sum(dim=1, keepdim=True)
    raise ValueError(f'invalid variance_type: {variance_type}')


class ProbabilisticEnsembleDynamicsModel(nn.Module):
    _models: nn.ModuleList

    def __init__(self, models: List[ProbabilisticDynamicsModel]):
        super().__init__()
        self._models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor, action: torch.Tensor, indices: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor]:
        return self.predict_with_variance(x, action, indices=indices)[:2]

    def __call__(self, x: torch.Tensor, action: torch.Tensor, indices: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor]:
        return cast(Tuple[torch.Tensor, torch.Tensor], super().__call__(x, action, indices))

    def predict_with_variance(self, x: torch.Tensor, action: torch.Tensor, variance_type: str='data', indices: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observations_list: List[torch.Tensor] = []
        rewards_list: List[torch.Tensor] = []
        variances_list: List[torch.Tensor] = []
        for model in self._models:
            obs, rew, var = model.predict_with_variance(x, action)
            observations_list.append(obs.view(1, x.shape[0], -1))
            rewards_list.append(rew.view(1, x.shape[0], 1))
            variances_list.append(var.view(1, x.shape[0], 1))
        observations = torch.cat(observations_list, dim=0).transpose(0, 1)
        rewards = torch.cat(rewards_list, dim=0).transpose(0, 1)
        variances = torch.cat(variances_list, dim=0).transpose(0, 1)
        variances = _compute_ensemble_variance(observations=observations, rewards=rewards, variances=variances, variance_type=variance_type)
        if indices is None:
            return observations, rewards, variances
        partial_observations = observations[torch.arange(x.shape[0]), indices]
        partial_rewards = rewards[torch.arange(x.shape[0]), indices]
        return partial_observations, partial_rewards, variances

    def compute_error(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, masks: Optional[torch.Tensor]=None) ->torch.Tensor:
        loss_sum = torch.tensor(0.0, dtype=torch.float32, device=observations.device)
        for i, model in enumerate(self._models):
            loss = model.compute_error(observations, actions, rewards, next_observations)
            assert loss.shape == (observations.shape[0], 1)
            if masks is None:
                mask = torch.randint(0, 2, size=loss.shape, device=observations.device)
            else:
                mask = masks[i]
            loss_sum += (loss * mask).mean()
        return loss_sum

    @property
    def models(self) ->nn.ModuleList:
        return self._models


class _PixelEncoder(nn.Module):
    _observation_shape: Sequence[int]
    _feature_size: int
    _use_batch_norm: bool
    _dropout_rate: Optional[float]
    _activation: nn.Module
    _convs: nn.ModuleList
    _conv_bns: nn.ModuleList
    _fc: nn.Linear
    _fc_bn: nn.BatchNorm1d
    _dropouts: nn.ModuleList

    def __init__(self, observation_shape: Sequence[int], filters: Optional[List[Sequence[int]]]=None, feature_size: int=512, use_batch_norm: bool=False, dropout_rate: Optional[float]=False, activation: nn.Module=nn.ReLU()):
        super().__init__()
        if filters is None:
            filters = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
        if feature_size is None:
            feature_size = 512
        self._observation_shape = observation_shape
        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate
        self._activation = activation
        self._feature_size = feature_size
        in_channels = [observation_shape[0]] + [f[0] for f in filters[:-1]]
        self._convs = nn.ModuleList()
        self._conv_bns = nn.ModuleList()
        self._dropouts = nn.ModuleList()
        for in_channel, f in zip(in_channels, filters):
            out_channel, kernel_size, stride = f
            conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
            self._convs.append(conv)
            if use_batch_norm:
                self._conv_bns.append(nn.BatchNorm2d(out_channel))
            if dropout_rate is not None:
                self._dropouts.append(nn.Dropout2d(dropout_rate))
        self._fc = nn.Linear(self._get_linear_input_size(), feature_size)
        if use_batch_norm:
            self._fc_bn = nn.BatchNorm1d(feature_size)
        if dropout_rate is not None:
            self._dropouts.append(nn.Dropout(dropout_rate))

    def _get_linear_input_size(self) ->int:
        x = torch.rand((1,) + tuple(self._observation_shape))
        with torch.no_grad():
            return self._conv_encode(x).view(1, -1).shape[1]

    def _get_last_conv_shape(self) ->Sequence[int]:
        x = torch.rand((1,) + tuple(self._observation_shape))
        with torch.no_grad():
            return self._conv_encode(x).shape

    def _conv_encode(self, x: torch.Tensor) ->torch.Tensor:
        h = x
        for i, conv in enumerate(self._convs):
            h = self._activation(conv(h))
            if self._use_batch_norm:
                h = self._conv_bns[i](h)
            if self._dropout_rate is not None:
                h = self._dropouts[i](h)
        return h

    def get_feature_size(self) ->int:
        return self._feature_size

    @property
    def observation_shape(self) ->Sequence[int]:
        return self._observation_shape

    @property
    def last_layer(self) ->nn.Linear:
        return self._fc


class Encoder(metaclass=ABCMeta):

    @abstractmethod
    def forward(self, x: torch.Tensor) ->torch.Tensor:
        pass

    @abstractmethod
    def get_feature_size(self) ->int:
        pass

    @property
    def observation_shape(self) ->Sequence[int]:
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor) ->torch.Tensor:
        pass

    @property
    def last_layer(self) ->nn.Linear:
        raise NotImplementedError


class PixelEncoder(_PixelEncoder, Encoder):

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        h = self._conv_encode(x)
        h = self._activation(self._fc(h.view(h.shape[0], -1)))
        if self._use_batch_norm:
            h = self._fc_bn(h)
        if self._dropout_rate is not None:
            h = self._dropouts[-1](h)
        return h


class PixelEncoderWithAction(_PixelEncoder, EncoderWithAction):
    _action_size: int
    _discrete_action: bool

    def __init__(self, observation_shape: Sequence[int], action_size: int, filters: Optional[List[Sequence[int]]]=None, feature_size: int=512, use_batch_norm: bool=False, dropout_rate: Optional[float]=None, discrete_action: bool=False, activation: nn.Module=nn.ReLU()):
        self._action_size = action_size
        self._discrete_action = discrete_action
        super().__init__(observation_shape=observation_shape, filters=filters, feature_size=feature_size, use_batch_norm=use_batch_norm, dropout_rate=dropout_rate, activation=activation)

    def _get_linear_input_size(self) ->int:
        size = super()._get_linear_input_size()
        return size + self._action_size

    def forward(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        h = self._conv_encode(x)
        if self._discrete_action:
            action = F.one_hot(action.view(-1).long(), num_classes=self._action_size).float()
        h = torch.cat([h.view(h.shape[0], -1), action], dim=1)
        h = self._activation(self._fc(h))
        if self._use_batch_norm:
            h = self._fc_bn(h)
        if self._dropout_rate is not None:
            h = self._dropouts[-1](h)
        return h

    @property
    def action_size(self) ->int:
        return self._action_size


class _VectorEncoder(nn.Module):
    _observation_shape: Sequence[int]
    _use_batch_norm: bool
    _dropout_rate: Optional[float]
    _use_dense: bool
    _activation: nn.Module
    _feature_size: int
    _fcs: nn.ModuleList
    _bns: nn.ModuleList
    _dropouts: nn.ModuleList

    def __init__(self, observation_shape: Sequence[int], hidden_units: Optional[Sequence[int]]=None, use_batch_norm: bool=False, dropout_rate: Optional[float]=None, use_dense: bool=False, activation: nn.Module=nn.ReLU()):
        super().__init__()
        self._observation_shape = observation_shape
        if hidden_units is None:
            hidden_units = [256, 256]
        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate
        self._feature_size = hidden_units[-1]
        self._activation = activation
        self._use_dense = use_dense
        in_units = [observation_shape[0]] + list(hidden_units[:-1])
        self._fcs = nn.ModuleList()
        self._bns = nn.ModuleList()
        self._dropouts = nn.ModuleList()
        for i, (in_unit, out_unit) in enumerate(zip(in_units, hidden_units)):
            if use_dense and i > 0:
                in_unit += observation_shape[0]
            self._fcs.append(nn.Linear(in_unit, out_unit))
            if use_batch_norm:
                self._bns.append(nn.BatchNorm1d(out_unit))
            if dropout_rate is not None:
                self._dropouts.append(nn.Dropout(dropout_rate))

    def _fc_encode(self, x: torch.Tensor) ->torch.Tensor:
        h = x
        for i, fc in enumerate(self._fcs):
            if self._use_dense and i > 0:
                h = torch.cat([h, x], dim=1)
            h = self._activation(fc(h))
            if self._use_batch_norm:
                h = self._bns[i](h)
            if self._dropout_rate is not None:
                h = self._dropouts[i](h)
        return h

    def get_feature_size(self) ->int:
        return self._feature_size

    @property
    def observation_shape(self) ->Sequence[int]:
        return self._observation_shape

    @property
    def last_layer(self) ->nn.Linear:
        return self._fcs[-1]


class VectorEncoder(_VectorEncoder, Encoder):

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        h = self._fc_encode(x)
        if self._use_batch_norm:
            h = self._bns[-1](h)
        if self._dropout_rate is not None:
            h = self._dropouts[-1](h)
        return h


class VectorEncoderWithAction(_VectorEncoder, EncoderWithAction):
    _action_size: int
    _discrete_action: bool

    def __init__(self, observation_shape: Sequence[int], action_size: int, hidden_units: Optional[Sequence[int]]=None, use_batch_norm: bool=False, dropout_rate: Optional[float]=None, use_dense: bool=False, discrete_action: bool=False, activation: nn.Module=nn.ReLU()):
        self._action_size = action_size
        self._discrete_action = discrete_action
        concat_shape = observation_shape[0] + action_size,
        super().__init__(observation_shape=concat_shape, hidden_units=hidden_units, use_batch_norm=use_batch_norm, use_dense=use_dense, dropout_rate=dropout_rate, activation=activation)
        self._observation_shape = observation_shape

    def forward(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        if self._discrete_action:
            action = F.one_hot(action.view(-1).long(), num_classes=self.action_size).float()
        x = torch.cat([x, action], dim=1)
        h = self._fc_encode(x)
        if self._use_batch_norm:
            h = self._bns[-1](h)
        if self._dropout_rate is not None:
            h = self._dropouts[-1](h)
        return h

    @property
    def action_size(self) ->int:
        return self._action_size


class ConditionalVAE(nn.Module):
    _encoder_encoder: EncoderWithAction
    _decoder_encoder: EncoderWithAction
    _beta: float
    _min_logstd: float
    _max_logstd: float
    _action_size: int
    _latent_size: int
    _mu: nn.Linear
    _logstd: nn.Linear
    _fc: nn.Linear

    def __init__(self, encoder_encoder: EncoderWithAction, decoder_encoder: EncoderWithAction, beta: float, min_logstd: float=-20.0, max_logstd: float=2.0):
        super().__init__()
        self._encoder_encoder = encoder_encoder
        self._decoder_encoder = decoder_encoder
        self._beta = beta
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd
        self._action_size = encoder_encoder.action_size
        self._latent_size = decoder_encoder.action_size
        self._mu = nn.Linear(encoder_encoder.get_feature_size(), self._latent_size)
        self._logstd = nn.Linear(encoder_encoder.get_feature_size(), self._latent_size)
        self._fc = nn.Linear(decoder_encoder.get_feature_size(), self._action_size)

    def forward(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        dist = self.encode(x, action)
        return self.decode(x, dist.rsample())

    def __call__(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action))

    def encode(self, x: torch.Tensor, action: torch.Tensor) ->Normal:
        h = self._encoder_encoder(x, action)
        mu = self._mu(h)
        logstd = self._logstd(h)
        clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return Normal(mu, clipped_logstd.exp())

    def decode(self, x: torch.Tensor, latent: torch.Tensor) ->torch.Tensor:
        h = self._decoder_encoder(x, latent)
        return torch.tanh(self._fc(h))

    def decode_without_squash(self, x: torch.Tensor, latent: torch.Tensor) ->torch.Tensor:
        h = self._decoder_encoder(x, latent)
        return self._fc(h)

    def compute_error(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        dist = self.encode(x, action)
        kl_loss = kl_divergence(dist, Normal(0.0, 1.0)).mean()
        y = self.decode(x, dist.rsample())
        return F.mse_loss(y, action) + cast(torch.Tensor, self._beta * kl_loss)

    def sample(self, x: torch.Tensor) ->torch.Tensor:
        latent = torch.randn((x.shape[0], self._latent_size), device=x.device)
        return self.decode(x, latent.clamp(-0.5, 0.5))

    def sample_n(self, x: torch.Tensor, n: int, with_squash: bool=True) ->torch.Tensor:
        flat_latent_shape = n * x.shape[0], self._latent_size
        flat_latent = torch.randn(flat_latent_shape, device=x.device)
        clipped_latent = flat_latent.clamp(-0.5, 0.5)
        repeated_x = x.expand((n, *x.shape))
        flat_x = repeated_x.reshape(-1, *x.shape[1:])
        if with_squash:
            flat_actions = self.decode(flat_x, clipped_latent)
        else:
            flat_actions = self.decode_without_squash(flat_x, clipped_latent)
        actions = flat_actions.view(n, x.shape[0], -1)
        return actions.transpose(0, 1)

    def sample_n_without_squash(self, x: torch.Tensor, n: int) ->torch.Tensor:
        return self.sample_n(x, n, with_squash=False)


class Imitator(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def forward(self, x: torch.Tensor) ->torch.Tensor:
        pass

    def __call__(self, x: torch.Tensor) ->torch.Tensor:
        return cast(torch.Tensor, super().__call__(x))

    @abstractmethod
    def compute_error(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        pass


class DiscreteImitator(Imitator):
    _encoder: Encoder
    _beta: float
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int, beta: float):
        super().__init__()
        self._encoder = encoder
        self._beta = beta
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return self.compute_log_probs_with_logits(x)[0]

    def compute_log_probs_with_logits(self, x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        h = self._encoder(x)
        logits = self._fc(h)
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs, logits

    def compute_error(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        log_probs, logits = self.compute_log_probs_with_logits(x)
        penalty = (logits ** 2).mean()
        return F.nll_loss(log_probs, action.view(-1)) + self._beta * penalty


class DeterministicRegressor(Imitator):
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        h = self._encoder(x)
        h = self._fc(h)
        return torch.tanh(h)

    def compute_error(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        return F.mse_loss(self.forward(x), action)


class ProbablisticRegressor(Imitator):
    _min_logstd: float
    _max_logstd: float
    _encoder: Encoder
    _mu: nn.Linear
    _logstd: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int, min_logstd: float, max_logstd: float):
        super().__init__()
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd
        self._encoder = encoder
        self._mu = nn.Linear(encoder.get_feature_size(), action_size)
        self._logstd = nn.Linear(encoder.get_feature_size(), action_size)

    def dist(self, x: torch.Tensor) ->Normal:
        h = self._encoder(x)
        mu = self._mu(h)
        logstd = self._logstd(h)
        clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return Normal(mu, clipped_logstd.exp())

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        h = self._encoder(x)
        mu = self._mu(h)
        return torch.tanh(mu)

    def sample_n(self, x: torch.Tensor, n: int) ->torch.Tensor:
        dist = self.dist(x)
        actions = cast(torch.Tensor, dist.rsample((n,)))
        return actions.transpose(0, 1)

    def compute_error(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        dist = self.dist(x)
        return F.mse_loss(torch.tanh(dist.rsample()), action)


class Parameter(nn.Module):
    _parameter: nn.Parameter

    def __init__(self, data: torch.Tensor):
        super().__init__()
        self._parameter = nn.Parameter(data)

    def forward(self) ->torch.Tensor:
        return self._parameter

    def __call__(self) ->torch.Tensor:
        return super().__call__()

    @property
    def data(self) ->torch.Tensor:
        return self._parameter.data


class Policy(nn.Module, metaclass=ABCMeta):

    def sample(self, x: torch.Tensor) ->torch.Tensor:
        return self.sample_with_log_prob(x)[0]

    @abstractmethod
    def sample_with_log_prob(self, x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        pass

    def sample_n(self, x: torch.Tensor, n: int) ->torch.Tensor:
        return self.sample_n_with_log_prob(x, n)[0]

    @abstractmethod
    def sample_n_with_log_prob(self, x: torch.Tensor, n: int) ->Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def best_action(self, x: torch.Tensor) ->torch.Tensor:
        pass


class DeterministicPolicy(Policy):
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        h = self._encoder(x)
        return torch.tanh(self._fc(h))

    def __call__(self, x: torch.Tensor) ->torch.Tensor:
        return cast(torch.Tensor, super().__call__(x))

    def sample_with_log_prob(self, x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('deterministic policy does not support sample')

    def sample_n_with_log_prob(self, x: torch.Tensor, n: int) ->Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('deterministic policy does not support sample_n')

    def best_action(self, x: torch.Tensor) ->torch.Tensor:
        return self.forward(x)


class DeterministicResidualPolicy(Policy):
    _encoder: EncoderWithAction
    _scale: float
    _fc: nn.Linear

    def __init__(self, encoder: EncoderWithAction, scale: float):
        super().__init__()
        self._scale = scale
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), encoder.action_size)

    def forward(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        h = self._encoder(x, action)
        residual_action = self._scale * torch.tanh(self._fc(h))
        return (action + cast(torch.Tensor, residual_action)).clamp(-1.0, 1.0)

    def __call__(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action))

    def best_residual_action(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        return self.forward(x, action)

    def best_action(self, x: torch.Tensor) ->torch.Tensor:
        raise NotImplementedError('residual policy does not support best_action')

    def sample_with_log_prob(self, x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('deterministic policy does not support sample')

    def sample_n_with_log_prob(self, x: torch.Tensor, n: int) ->Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('deterministic policy does not support sample_n')


class Distribution(metaclass=ABCMeta):

    @abstractmethod
    def sample(self) ->torch.Tensor:
        pass

    @abstractmethod
    def sample_with_log_prob(self) ->Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def sample_n(self, n: int) ->torch.Tensor:
        pass

    @abstractmethod
    def sample_n_with_log_prob(self, n: int) ->Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def log_prob(self, y: torch.Tensor) ->torch.Tensor:
        pass


class GaussianDistribution(Distribution):
    _raw_loc: torch.Tensor
    _mean: torch.Tensor
    _std: torch.Tensor
    _dist: Normal

    def __init__(self, loc: torch.Tensor, std: torch.Tensor, raw_loc: Optional[torch.Tensor]=None):
        self._mean = loc
        self._std = std
        if raw_loc is not None:
            self._raw_loc = raw_loc
        self._dist = Normal(self._mean, self._std)

    def sample(self) ->torch.Tensor:
        return self._dist.rsample().clamp(-1.0, 1.0)

    def sample_with_log_prob(self) ->Tuple[torch.Tensor, torch.Tensor]:
        y = self.sample()
        return y, self.log_prob(y)

    def sample_without_squash(self) ->torch.Tensor:
        assert self._raw_loc is not None
        return Normal(self._raw_loc, self._std).rsample()

    def sample_n(self, n: int) ->torch.Tensor:
        return self._dist.rsample((n,)).clamp(-1.0, 1.0)

    def sample_n_with_log_prob(self, n: int) ->Tuple[torch.Tensor, torch.Tensor]:
        x = self.sample_n(n)
        return x, self.log_prob(x)

    def sample_n_without_squash(self, n: int) ->torch.Tensor:
        assert self._raw_loc is not None
        return Normal(self._raw_loc, self._std).rsample((n,))

    def mean_with_log_prob(self) ->Tuple[torch.Tensor, torch.Tensor]:
        return self._mean, self.log_prob(self._mean)

    def log_prob(self, y: torch.Tensor) ->torch.Tensor:
        return self._dist.log_prob(y).sum(dim=-1, keepdims=True)

    @property
    def mean(self) ->torch.Tensor:
        return self._mean

    @property
    def std(self) ->torch.Tensor:
        return self._std


class SquashedGaussianDistribution(Distribution):
    _mean: torch.Tensor
    _std: torch.Tensor
    _dist: Normal

    def __init__(self, loc: torch.Tensor, std: torch.Tensor):
        self._mean = loc
        self._std = std
        self._dist = Normal(self._mean, self._std)

    def sample(self) ->torch.Tensor:
        return torch.tanh(self._dist.rsample())

    def sample_with_log_prob(self) ->Tuple[torch.Tensor, torch.Tensor]:
        raw_y = self._dist.rsample()
        log_prob = self._log_prob_from_raw_y(raw_y)
        return torch.tanh(raw_y), log_prob

    def sample_without_squash(self) ->torch.Tensor:
        return self._dist.rsample()

    def sample_n(self, n: int) ->torch.Tensor:
        return torch.tanh(self._dist.rsample((n,)))

    def sample_n_with_log_prob(self, n: int) ->Tuple[torch.Tensor, torch.Tensor]:
        raw_y = self._dist.rsample((n,))
        log_prob = self._log_prob_from_raw_y(raw_y)
        return torch.tanh(raw_y), log_prob

    def sample_n_without_squash(self, n: int) ->torch.Tensor:
        return self._dist.rsample((n,))

    def mean_with_log_prob(self) ->Tuple[torch.Tensor, torch.Tensor]:
        return torch.tanh(self._mean), self._log_prob_from_raw_y(self._mean)

    def log_prob(self, y: torch.Tensor) ->torch.Tensor:
        clipped_y = y.clamp(-0.999999, 0.999999)
        raw_y = torch.atanh(clipped_y)
        return self._log_prob_from_raw_y(raw_y)

    def _log_prob_from_raw_y(self, raw_y: torch.Tensor) ->torch.Tensor:
        jacob = 2 * (math.log(2) - raw_y - F.softplus(-2 * raw_y))
        return (self._dist.log_prob(raw_y) - jacob).sum(dim=-1, keepdims=True)

    @property
    def mean(self) ->torch.Tensor:
        return torch.tanh(self._mean)

    @property
    def std(self) ->torch.Tensor:
        return self._std


class NormalPolicy(Policy):
    _encoder: Encoder
    _action_size: int
    _min_logstd: float
    _max_logstd: float
    _use_std_parameter: bool
    _mu: nn.Linear
    _logstd: Union[nn.Linear, nn.Parameter]

    def __init__(self, encoder: Encoder, action_size: int, min_logstd: float, max_logstd: float, use_std_parameter: bool, squash_distribution: bool):
        super().__init__()
        self._action_size = action_size
        self._encoder = encoder
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd
        self._use_std_parameter = use_std_parameter
        self._squash_distribution = squash_distribution
        self._mu = nn.Linear(encoder.get_feature_size(), action_size)
        if use_std_parameter:
            initial_logstd = torch.zeros(1, action_size, dtype=torch.float32)
            self._logstd = nn.Parameter(initial_logstd)
        else:
            self._logstd = nn.Linear(encoder.get_feature_size(), action_size)

    def _compute_logstd(self, h: torch.Tensor) ->torch.Tensor:
        if self._use_std_parameter:
            clipped_logstd = self.get_logstd_parameter()
        else:
            logstd = cast(nn.Linear, self._logstd)(h)
            clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return clipped_logstd

    def dist(self, x: torch.Tensor) ->Union[GaussianDistribution, SquashedGaussianDistribution]:
        h = self._encoder(x)
        mu = self._mu(h)
        clipped_logstd = self._compute_logstd(h)
        if self._squash_distribution:
            return SquashedGaussianDistribution(mu, clipped_logstd.exp())
        else:
            return GaussianDistribution(torch.tanh(mu), clipped_logstd.exp(), raw_loc=mu)

    def forward(self, x: torch.Tensor, deterministic: bool=False, with_log_prob: bool=False) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dist = self.dist(x)
        if deterministic:
            action, log_prob = dist.mean_with_log_prob()
        else:
            action, log_prob = dist.sample_with_log_prob()
        return (action, log_prob) if with_log_prob else action

    def sample_with_log_prob(self, x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(x, with_log_prob=True)
        return cast(Tuple[torch.Tensor, torch.Tensor], out)

    def sample_n_with_log_prob(self, x: torch.Tensor, n: int) ->Tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist(x)
        action_T, log_prob_T = dist.sample_n_with_log_prob(n)
        transposed_action = action_T.transpose(0, 1)
        log_prob = log_prob_T.transpose(0, 1)
        return transposed_action, log_prob

    def sample_n_without_squash(self, x: torch.Tensor, n: int) ->torch.Tensor:
        dist = self.dist(x)
        action = dist.sample_n_without_squash(n)
        return action.transpose(0, 1)

    def onnx_safe_sample_n(self, x: torch.Tensor, n: int) ->torch.Tensor:
        h = self._encoder(x)
        mean = self._mu(h)
        std = self._compute_logstd(h).exp()
        if not self._squash_distribution:
            mean = torch.tanh(mean)
        expanded_mean = mean.view(-1, 1, self._action_size).repeat((1, n, 1))
        expanded_std = std.view(-1, 1, self._action_size).repeat((1, n, 1))
        noise = torch.randn(x.shape[0], n, self._action_size, device=x.device)
        if self._squash_distribution:
            return torch.tanh(expanded_mean + noise * expanded_std)
        else:
            return expanded_mean + noise * expanded_std

    def best_action(self, x: torch.Tensor) ->torch.Tensor:
        action = self.forward(x, deterministic=True, with_log_prob=False)
        return cast(torch.Tensor, action)

    def get_logstd_parameter(self) ->torch.Tensor:
        assert self._use_std_parameter
        logstd = torch.sigmoid(cast(nn.Parameter, self._logstd))
        base_logstd = self._max_logstd - self._min_logstd
        return self._min_logstd + logstd * base_logstd


class SquashedNormalPolicy(NormalPolicy):

    def __init__(self, encoder: Encoder, action_size: int, min_logstd: float, max_logstd: float, use_std_parameter: bool):
        super().__init__(encoder=encoder, action_size=action_size, min_logstd=min_logstd, max_logstd=max_logstd, use_std_parameter=use_std_parameter, squash_distribution=True)


class NonSquashedNormalPolicy(NormalPolicy):

    def __init__(self, encoder: Encoder, action_size: int, min_logstd: float, max_logstd: float, use_std_parameter: bool):
        super().__init__(encoder=encoder, action_size=action_size, min_logstd=min_logstd, max_logstd=max_logstd, use_std_parameter=use_std_parameter, squash_distribution=False)


class CategoricalPolicy(Policy):
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)

    def dist(self, x: torch.Tensor) ->Categorical:
        h = self._encoder(x)
        h = self._fc(h)
        return Categorical(torch.softmax(h, dim=1))

    def forward(self, x: torch.Tensor, deterministic: bool=False, with_log_prob: bool=False) ->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dist = self.dist(x)
        if deterministic:
            action = cast(torch.Tensor, dist.probs.argmax(dim=1))
        else:
            action = cast(torch.Tensor, dist.sample())
        if with_log_prob:
            return action, dist.log_prob(action)
        return action

    def sample_with_log_prob(self, x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(x, with_log_prob=True)
        return cast(Tuple[torch.Tensor, torch.Tensor], out)

    def sample_n_with_log_prob(self, x: torch.Tensor, n: int) ->Tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist(x)
        action_T = cast(torch.Tensor, dist.sample((n,)))
        log_prob_T = dist.log_prob(action_T)
        action = action_T.transpose(0, 1)
        log_prob = log_prob_T.transpose(0, 1)
        return action, log_prob

    def best_action(self, x: torch.Tensor) ->torch.Tensor:
        return cast(torch.Tensor, self.forward(x, deterministic=True))

    def log_probs(self, x: torch.Tensor) ->torch.Tensor:
        dist = self.dist(x)
        return cast(torch.Tensor, dist.logits)


class QFunction(metaclass=ABCMeta):

    @abstractmethod
    def compute_error(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, target: torch.Tensor, terminals: torch.Tensor, gamma: float=0.99, reduction: str='mean') ->torch.Tensor:
        pass

    @property
    def action_size(self) ->int:
        pass


class ContinuousQFunction(QFunction):

    @abstractmethod
    def forward(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        pass

    @abstractmethod
    def compute_target(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        pass

    def __call__(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        return self.forward(x, action)

    @property
    def encoder(self) ->EncoderWithAction:
        pass


class DiscreteQFunction(QFunction):

    @abstractmethod
    def forward(self, x: torch.Tensor) ->torch.Tensor:
        pass

    @abstractmethod
    def compute_target(self, x: torch.Tensor, action: Optional[torch.Tensor]) ->torch.Tensor:
        pass

    def __call__(self, x: torch.Tensor) ->torch.Tensor:
        return self.forward(x)

    @property
    def encoder(self) ->Encoder:
        pass


def _reduce_ensemble(y: torch.Tensor, reduction: str='min', dim: int=0, lam: float=0.75) ->torch.Tensor:
    if reduction == 'min':
        return y.min(dim=dim).values
    elif reduction == 'max':
        return y.max(dim=dim).values
    elif reduction == 'mean':
        return y.mean(dim=dim)
    elif reduction == 'none':
        return y
    elif reduction == 'mix':
        max_values = y.max(dim=dim).values
        min_values = y.min(dim=dim).values
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


def _gather_quantiles_by_indices(y: torch.Tensor, indices: torch.Tensor) ->torch.Tensor:
    if y.dim() == 3:
        return y.transpose(0, 1)[torch.arange(y.shape[1]), indices]
    elif y.dim() == 4:
        transposed_y = y.transpose(0, 1).transpose(1, 2)
        flat_y = transposed_y.reshape(-1, y.shape[0], y.shape[3])
        head_indices = torch.arange(y.shape[1] * y.shape[2])
        gathered_y = flat_y[head_indices, indices.view(-1)]
        return gathered_y.view(y.shape[1], y.shape[2], -1)
    raise ValueError


def _reduce_quantile_ensemble(y: torch.Tensor, reduction: str='min', dim: int=0, lam: float=0.75) ->torch.Tensor:
    mean = y.mean(dim=-1)
    if reduction == 'min':
        indices = mean.min(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == 'max':
        indices = mean.max(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == 'none':
        return y
    elif reduction == 'mix':
        min_indices = mean.min(dim=dim).indices
        max_indices = mean.max(dim=dim).indices
        min_values = _gather_quantiles_by_indices(y, min_indices)
        max_values = _gather_quantiles_by_indices(y, max_indices)
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


class EnsembleQFunction(nn.Module):
    _action_size: int
    _q_funcs: nn.ModuleList

    def __init__(self, q_funcs: Union[List[DiscreteQFunction], List[ContinuousQFunction]]):
        super().__init__()
        self._action_size = q_funcs[0].action_size
        self._q_funcs = nn.ModuleList(q_funcs)

    def compute_error(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, target: torch.Tensor, terminals: torch.Tensor, gamma: float=0.99) ->torch.Tensor:
        assert target.ndim == 2
        td_sum = torch.tensor(0.0, dtype=torch.float32, device=observations.device)
        for q_func in self._q_funcs:
            loss = q_func.compute_error(observations=observations, actions=actions, rewards=rewards, target=target, terminals=terminals, gamma=gamma, reduction='none')
            td_sum += loss.mean()
        return td_sum

    def _compute_target(self, x: torch.Tensor, action: Optional[torch.Tensor]=None, reduction: str='min', lam: float=0.75) ->torch.Tensor:
        values_list: List[torch.Tensor] = []
        for q_func in self._q_funcs:
            target = q_func.compute_target(x, action)
            values_list.append(target.reshape(1, x.shape[0], -1))
        values = torch.cat(values_list, dim=0)
        if action is None:
            if values.shape[2] == self._action_size:
                return _reduce_ensemble(values, reduction)
            n_q_funcs = values.shape[0]
            values = values.view(n_q_funcs, x.shape[0], self._action_size, -1)
            return _reduce_quantile_ensemble(values, reduction)
        if values.shape[2] == 1:
            return _reduce_ensemble(values, reduction, lam=lam)
        return _reduce_quantile_ensemble(values, reduction, lam=lam)

    @property
    def q_funcs(self) ->nn.ModuleList:
        return self._q_funcs


class EnsembleDiscreteQFunction(EnsembleQFunction):

    def forward(self, x: torch.Tensor, reduction: str='mean') ->torch.Tensor:
        values = []
        for q_func in self._q_funcs:
            values.append(q_func(x).view(1, x.shape[0], self._action_size))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def __call__(self, x: torch.Tensor, reduction: str='mean') ->torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, reduction))

    def compute_target(self, x: torch.Tensor, action: Optional[torch.Tensor]=None, reduction: str='min', lam: float=0.75) ->torch.Tensor:
        return self._compute_target(x, action, reduction, lam)


class EnsembleContinuousQFunction(EnsembleQFunction):

    def forward(self, x: torch.Tensor, action: torch.Tensor, reduction: str='mean') ->torch.Tensor:
        values = []
        for q_func in self._q_funcs:
            values.append(q_func(x, action).view(1, x.shape[0], 1))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def __call__(self, x: torch.Tensor, action: torch.Tensor, reduction: str='mean') ->torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action, reduction))

    def compute_target(self, x: torch.Tensor, action: torch.Tensor, reduction: str='min', lam: float=0.75) ->torch.Tensor:
        return self._compute_target(x, action, reduction, lam)


def _make_taus(h: torch.Tensor, n_quantiles: int) ->torch.Tensor:
    steps = torch.arange(n_quantiles, dtype=torch.float32, device=h.device)
    taus = ((steps + 1).float() / n_quantiles).view(1, -1)
    taus_dot = (steps.float() / n_quantiles).view(1, -1)
    return (taus + taus_dot) / 2.0


def compute_iqn_feature(h: torch.Tensor, taus: torch.Tensor, embed: nn.Linear, embed_size: int) ->torch.Tensor:
    steps = torch.arange(embed_size, device=h.device).float() + 1
    expanded_taus = taus.view(h.shape[0], -1, 1)
    prior = torch.cos(math.pi * steps.view(1, 1, -1) * expanded_taus)
    phi = torch.relu(embed(prior))
    return h.view(h.shape[0], 1, -1) * phi


def compute_huber_loss(y: torch.Tensor, target: torch.Tensor, beta: float=1.0) ->torch.Tensor:
    diff = target - y
    cond = diff.detach().abs() < beta
    return torch.where(cond, 0.5 * diff ** 2, beta * (diff.abs() - 0.5 * beta))


def compute_quantile_huber_loss(y: torch.Tensor, target: torch.Tensor, taus: torch.Tensor) ->torch.Tensor:
    assert y.dim() == 3 and target.dim() == 3 and taus.dim() == 3
    huber_loss = compute_huber_loss(y, target)
    delta = cast(torch.Tensor, ((target - y).detach() < 0.0).float())
    element_wise_loss = (taus - delta).abs() * huber_loss
    return element_wise_loss.sum(dim=2).mean(dim=1)


def compute_quantile_loss(quantiles: torch.Tensor, rewards: torch.Tensor, target: torch.Tensor, terminals: torch.Tensor, taus: torch.Tensor, gamma: float) ->torch.Tensor:
    batch_size, n_quantiles = quantiles.shape
    expanded_quantiles = quantiles.view(batch_size, 1, -1)
    y = rewards + gamma * target * (1 - terminals)
    expanded_y = y.view(batch_size, -1, 1)
    expanded_taus = taus.view(-1, 1, n_quantiles)
    return compute_quantile_huber_loss(expanded_quantiles, expanded_y, expanded_taus)


def compute_reduce(value: torch.Tensor, reduction_type: str) ->torch.Tensor:
    if reduction_type == 'mean':
        return value.mean()
    elif reduction_type == 'sum':
        return value.sum()
    elif reduction_type == 'none':
        return value.view(-1, 1)
    raise ValueError('invalid reduction type.')


def pick_quantile_value_by_action(values: torch.Tensor, action: torch.Tensor, keepdim: bool=False) ->torch.Tensor:
    assert values.ndim == 3
    action_size = values.shape[1]
    one_hot = F.one_hot(action.view(-1), num_classes=action_size)
    mask = cast(torch.Tensor, one_hot.view(-1, action_size, 1).float())
    return (values * mask).sum(dim=1, keepdim=keepdim)


class DiscreteFQFQFunction(DiscreteQFunction, nn.Module):
    _action_size: int
    _entropy_coeff: float
    _encoder: Encoder
    _fc: nn.Linear
    _n_quantiles: int
    _embed_size: int
    _embed: nn.Linear
    _proposal: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int, n_quantiles: int, embed_size: int, entropy_coeff: float=0.0):
        super().__init__()
        self._encoder = encoder
        self._action_size = action_size
        self._fc = nn.Linear(encoder.get_feature_size(), self._action_size)
        self._entropy_coeff = entropy_coeff
        self._n_quantiles = n_quantiles
        self._embed_size = embed_size
        self._embed = nn.Linear(embed_size, encoder.get_feature_size())
        self._proposal = nn.Linear(encoder.get_feature_size(), n_quantiles)

    def _compute_quantiles(self, h: torch.Tensor, taus: torch.Tensor) ->torch.Tensor:
        prod = compute_iqn_feature(h, taus, self._embed, self._embed_size)
        return cast(torch.Tensor, self._fc(prod)).transpose(1, 2)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        h = self._encoder(x)
        taus, taus_minus, taus_prime, _ = _make_taus(h, self._proposal)
        quantiles = self._compute_quantiles(h, taus_prime.detach())
        weight = (taus - taus_minus).view(-1, 1, self._n_quantiles).detach()
        return (weight * quantiles).sum(dim=2)

    def compute_error(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, target: torch.Tensor, terminals: torch.Tensor, gamma: float=0.99, reduction: str='mean') ->torch.Tensor:
        assert target.shape == (observations.shape[0], self._n_quantiles)
        h = self._encoder(observations)
        taus, _, taus_prime, entropies = _make_taus(h, self._proposal)
        all_quantiles = self._compute_quantiles(h, taus_prime.detach())
        quantiles = pick_quantile_value_by_action(all_quantiles, actions)
        quantile_loss = compute_quantile_loss(quantiles=quantiles, rewards=rewards, target=target, terminals=terminals, taus=taus_prime.detach(), gamma=gamma)
        proposal_loss = self._compute_proposal_loss(h, actions, taus, taus_prime)
        proposal_params = list(self._proposal.parameters())
        proposal_grads = torch.autograd.grad(outputs=proposal_loss.mean(), inputs=proposal_params, retain_graph=True)
        for param, grad in zip(list(proposal_params), proposal_grads):
            param.grad = 0.0001 * grad
        loss = quantile_loss - self._entropy_coeff * entropies
        return compute_reduce(loss, reduction)

    def _compute_proposal_loss(self, h: torch.Tensor, actions: torch.Tensor, taus: torch.Tensor, taus_prime: torch.Tensor) ->torch.Tensor:
        q_taus = self._compute_quantiles(h.detach(), taus)
        q_taus_prime = self._compute_quantiles(h.detach(), taus_prime)
        batch_steps = torch.arange(h.shape[0])
        q_taus = q_taus[batch_steps, actions.view(-1)][:, :-1]
        q_taus_prime = q_taus_prime[batch_steps, actions.view(-1)]
        proposal_grad = 2 * q_taus - q_taus_prime[:, :-1] - q_taus_prime[:, 1:]
        return proposal_grad.sum(dim=1)

    def compute_target(self, x: torch.Tensor, action: Optional[torch.Tensor]=None) ->torch.Tensor:
        h = self._encoder(x)
        _, _, taus_prime, _ = _make_taus(h, self._proposal)
        quantiles = self._compute_quantiles(h, taus_prime.detach())
        if action is None:
            return quantiles
        return pick_quantile_value_by_action(quantiles, action)

    @property
    def action_size(self) ->int:
        return self._action_size

    @property
    def encoder(self) ->Encoder:
        return self._encoder


class ContinuousFQFQFunction(ContinuousQFunction, nn.Module):
    _action_size: int
    _entropy_coeff: float
    _encoder: EncoderWithAction
    _fc: nn.Linear
    _n_quantiles: int
    _embed_size: int
    _embed: nn.Linear
    _proposal: nn.Linear

    def __init__(self, encoder: EncoderWithAction, n_quantiles: int, embed_size: int, entropy_coeff: float=0.0):
        super().__init__()
        self._encoder = encoder
        self._action_size = encoder.action_size
        self._fc = nn.Linear(encoder.get_feature_size(), 1)
        self._entropy_coeff = entropy_coeff
        self._n_quantiles = n_quantiles
        self._embed_size = embed_size
        self._embed = nn.Linear(embed_size, encoder.get_feature_size())
        self._proposal = nn.Linear(encoder.get_feature_size(), n_quantiles)

    def _compute_quantiles(self, h: torch.Tensor, taus: torch.Tensor) ->torch.Tensor:
        prod = compute_iqn_feature(h, taus, self._embed, self._embed_size)
        return cast(torch.Tensor, self._fc(prod)).view(h.shape[0], -1)

    def forward(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        h = self._encoder(x, action)
        taus, taus_minus, taus_prime, _ = _make_taus(h, self._proposal)
        quantiles = self._compute_quantiles(h, taus_prime.detach())
        weight = (taus - taus_minus).detach()
        return (weight * quantiles).sum(dim=1, keepdim=True)

    def compute_error(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, target: torch.Tensor, terminals: torch.Tensor, gamma: float=0.99, reduction: str='mean') ->torch.Tensor:
        assert target.shape == (observations.shape[0], self._n_quantiles)
        h = self._encoder(observations, actions)
        taus, _, taus_prime, entropies = _make_taus(h, self._proposal)
        quantiles = self._compute_quantiles(h, taus_prime.detach())
        quantile_loss = compute_quantile_loss(quantiles=quantiles, rewards=rewards, target=target, terminals=terminals, taus=taus_prime.detach(), gamma=gamma)
        proposal_loss = self._compute_proposal_loss(h, taus, taus_prime)
        proposal_params = list(self._proposal.parameters())
        proposal_grads = torch.autograd.grad(outputs=proposal_loss.mean(), inputs=proposal_params, retain_graph=True)
        for param, grad in zip(list(proposal_params), proposal_grads):
            param.grad = 0.0001 * grad
        loss = quantile_loss - self._entropy_coeff * entropies
        return compute_reduce(loss, reduction)

    def _compute_proposal_loss(self, h: torch.Tensor, taus: torch.Tensor, taus_prime: torch.Tensor) ->torch.Tensor:
        q_taus = self._compute_quantiles(h.detach(), taus)[:, :-1]
        q_taus_prime = self._compute_quantiles(h.detach(), taus_prime)
        proposal_grad = 2 * q_taus - q_taus_prime[:, :-1] - q_taus_prime[:, 1:]
        return proposal_grad.sum(dim=1)

    def compute_target(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        h = self._encoder(x, action)
        _, _, taus_prime, _ = _make_taus(h, self._proposal)
        return self._compute_quantiles(h, taus_prime.detach())

    @property
    def action_size(self) ->int:
        return self._action_size

    @property
    def encoder(self) ->EncoderWithAction:
        return self._encoder


class DiscreteIQNQFunction(DiscreteQFunction, nn.Module):
    _action_size: int
    _encoder: Encoder
    _fc: nn.Linear
    _n_quantiles: int
    _n_greedy_quantiles: int
    _embed_size: int
    _embed: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int, n_quantiles: int, n_greedy_quantiles: int, embed_size: int):
        super().__init__()
        self._encoder = encoder
        self._action_size = action_size
        self._fc = nn.Linear(encoder.get_feature_size(), self._action_size)
        self._n_quantiles = n_quantiles
        self._n_greedy_quantiles = n_greedy_quantiles
        self._embed_size = embed_size
        self._embed = nn.Linear(embed_size, encoder.get_feature_size())

    def _make_taus(self, h: torch.Tensor) ->torch.Tensor:
        if self.training:
            n_quantiles = self._n_quantiles
        else:
            n_quantiles = self._n_greedy_quantiles
        return _make_taus(h, n_quantiles, self.training)

    def _compute_quantiles(self, h: torch.Tensor, taus: torch.Tensor) ->torch.Tensor:
        prod = compute_iqn_feature(h, taus, self._embed, self._embed_size)
        return cast(torch.Tensor, self._fc(prod)).transpose(1, 2)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        h = self._encoder(x)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        return quantiles.mean(dim=2)

    def compute_error(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, target: torch.Tensor, terminals: torch.Tensor, gamma: float=0.99, reduction: str='mean') ->torch.Tensor:
        assert target.shape == (observations.shape[0], self._n_quantiles)
        h = self._encoder(observations)
        taus = self._make_taus(h)
        all_quantiles = self._compute_quantiles(h, taus)
        quantiles = pick_quantile_value_by_action(all_quantiles, actions)
        loss = compute_quantile_loss(quantiles=quantiles, rewards=rewards, target=target, terminals=terminals, taus=taus, gamma=gamma)
        return compute_reduce(loss, reduction)

    def compute_target(self, x: torch.Tensor, action: Optional[torch.Tensor]=None) ->torch.Tensor:
        h = self._encoder(x)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        if action is None:
            return quantiles
        return pick_quantile_value_by_action(quantiles, action)

    @property
    def action_size(self) ->int:
        return self._action_size

    @property
    def encoder(self) ->Encoder:
        return self._encoder


class ContinuousIQNQFunction(ContinuousQFunction, nn.Module):
    _action_size: int
    _encoder: EncoderWithAction
    _fc: nn.Linear
    _n_quantiles: int
    _n_greedy_quantiles: int
    _embed_size: int
    _embed: nn.Linear

    def __init__(self, encoder: EncoderWithAction, n_quantiles: int, n_greedy_quantiles: int, embed_size: int):
        super().__init__()
        self._encoder = encoder
        self._action_size = encoder.action_size
        self._fc = nn.Linear(encoder.get_feature_size(), 1)
        self._n_quantiles = n_quantiles
        self._n_greedy_quantiles = n_greedy_quantiles
        self._embed_size = embed_size
        self._embed = nn.Linear(embed_size, encoder.get_feature_size())

    def _make_taus(self, h: torch.Tensor) ->torch.Tensor:
        if self.training:
            n_quantiles = self._n_quantiles
        else:
            n_quantiles = self._n_greedy_quantiles
        return _make_taus(h, n_quantiles, self.training)

    def _compute_quantiles(self, h: torch.Tensor, taus: torch.Tensor) ->torch.Tensor:
        prod = compute_iqn_feature(h, taus, self._embed, self._embed_size)
        return cast(torch.Tensor, self._fc(prod)).view(h.shape[0], -1)

    def forward(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        h = self._encoder(x, action)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        return quantiles.mean(dim=1, keepdim=True)

    def compute_error(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, target: torch.Tensor, terminals: torch.Tensor, gamma: float=0.99, reduction: str='mean') ->torch.Tensor:
        assert target.shape == (observations.shape[0], self._n_quantiles)
        h = self._encoder(observations, actions)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        loss = compute_quantile_loss(quantiles=quantiles, rewards=rewards, target=target, terminals=terminals, taus=taus, gamma=gamma)
        return compute_reduce(loss, reduction)

    def compute_target(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        h = self._encoder(x, action)
        taus = self._make_taus(h)
        return self._compute_quantiles(h, taus)

    @property
    def action_size(self) ->int:
        return self._action_size

    @property
    def encoder(self) ->EncoderWithAction:
        return self._encoder


def pick_value_by_action(values: torch.Tensor, action: torch.Tensor, keepdim: bool=False) ->torch.Tensor:
    assert values.ndim == 2
    action_size = values.shape[1]
    one_hot = F.one_hot(action.view(-1), num_classes=action_size)
    masked_values = values * cast(torch.Tensor, one_hot.float())
    return masked_values.sum(dim=1, keepdim=keepdim)


class DiscreteMeanQFunction(DiscreteQFunction, nn.Module):
    _action_size: int
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int):
        super().__init__()
        self._action_size = action_size
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), action_size)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return cast(torch.Tensor, self._fc(self._encoder(x)))

    def compute_error(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, target: torch.Tensor, terminals: torch.Tensor, gamma: float=0.99, reduction: str='mean') ->torch.Tensor:
        one_hot = F.one_hot(actions.view(-1), num_classes=self.action_size)
        value = (self.forward(observations) * one_hot.float()).sum(dim=1, keepdim=True)
        y = rewards + gamma * target * (1 - terminals)
        loss = compute_huber_loss(value, y)
        return compute_reduce(loss, reduction)

    def compute_target(self, x: torch.Tensor, action: Optional[torch.Tensor]=None) ->torch.Tensor:
        if action is None:
            return self.forward(x)
        return pick_value_by_action(self.forward(x), action, keepdim=True)

    @property
    def action_size(self) ->int:
        return self._action_size

    @property
    def encoder(self) ->Encoder:
        return self._encoder


class ContinuousMeanQFunction(ContinuousQFunction, nn.Module):
    _encoder: EncoderWithAction
    _action_size: int
    _fc: nn.Linear

    def __init__(self, encoder: EncoderWithAction):
        super().__init__()
        self._encoder = encoder
        self._action_size = encoder.action_size
        self._fc = nn.Linear(encoder.get_feature_size(), 1)

    def forward(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        return cast(torch.Tensor, self._fc(self._encoder(x, action)))

    def compute_error(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, target: torch.Tensor, terminals: torch.Tensor, gamma: float=0.99, reduction: str='mean') ->torch.Tensor:
        value = self.forward(observations, actions)
        y = rewards + gamma * target * (1 - terminals)
        loss = F.mse_loss(value, y, reduction='none')
        return compute_reduce(loss, reduction)

    def compute_target(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        return self.forward(x, action)

    @property
    def action_size(self) ->int:
        return self._action_size

    @property
    def encoder(self) ->EncoderWithAction:
        return self._encoder


class DiscreteQRQFunction(DiscreteQFunction, nn.Module):
    _action_size: int
    _encoder: Encoder
    _n_quantiles: int
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int, n_quantiles: int):
        super().__init__()
        self._encoder = encoder
        self._action_size = action_size
        self._n_quantiles = n_quantiles
        self._fc = nn.Linear(encoder.get_feature_size(), action_size * n_quantiles)

    def _compute_quantiles(self, h: torch.Tensor, taus: torch.Tensor) ->torch.Tensor:
        h = cast(torch.Tensor, self._fc(h))
        return h.view(-1, self._action_size, self._n_quantiles)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        h = self._encoder(x)
        taus = _make_taus(h, self._n_quantiles)
        quantiles = self._compute_quantiles(h, taus)
        return quantiles.mean(dim=2)

    def compute_error(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, target: torch.Tensor, terminals: torch.Tensor, gamma: float=0.99, reduction: str='mean') ->torch.Tensor:
        assert target.shape == (observations.shape[0], self._n_quantiles)
        h = self._encoder(observations)
        taus = _make_taus(h, self._n_quantiles)
        all_quantiles = self._compute_quantiles(h, taus)
        quantiles = pick_quantile_value_by_action(all_quantiles, actions)
        loss = compute_quantile_loss(quantiles=quantiles, rewards=rewards, target=target, terminals=terminals, taus=taus, gamma=gamma)
        return compute_reduce(loss, reduction)

    def compute_target(self, x: torch.Tensor, action: Optional[torch.Tensor]=None) ->torch.Tensor:
        h = self._encoder(x)
        taus = _make_taus(h, self._n_quantiles)
        quantiles = self._compute_quantiles(h, taus)
        if action is None:
            return quantiles
        return pick_quantile_value_by_action(quantiles, action)

    @property
    def action_size(self) ->int:
        return self._action_size

    @property
    def encoder(self) ->Encoder:
        return self._encoder


class ContinuousQRQFunction(ContinuousQFunction, nn.Module):
    _action_size: int
    _encoder: EncoderWithAction
    _n_quantiles: int
    _fc: nn.Linear

    def __init__(self, encoder: EncoderWithAction, n_quantiles: int):
        super().__init__()
        self._encoder = encoder
        self._action_size = encoder.action_size
        self._n_quantiles = n_quantiles
        self._fc = nn.Linear(encoder.get_feature_size(), n_quantiles)

    def _compute_quantiles(self, h: torch.Tensor, taus: torch.Tensor) ->torch.Tensor:
        return cast(torch.Tensor, self._fc(h))

    def forward(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        h = self._encoder(x, action)
        taus = _make_taus(h, self._n_quantiles)
        quantiles = self._compute_quantiles(h, taus)
        return quantiles.mean(dim=1, keepdim=True)

    def compute_error(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, target: torch.Tensor, terminals: torch.Tensor, gamma: float=0.99, reduction: str='mean') ->torch.Tensor:
        assert target.shape == (observations.shape[0], self._n_quantiles)
        h = self._encoder(observations, actions)
        taus = _make_taus(h, self._n_quantiles)
        quantiles = self._compute_quantiles(h, taus)
        loss = compute_quantile_loss(quantiles=quantiles, rewards=rewards, target=target, terminals=terminals, taus=taus, gamma=gamma)
        return compute_reduce(loss, reduction)

    def compute_target(self, x: torch.Tensor, action: torch.Tensor) ->torch.Tensor:
        h = self._encoder(x, action)
        taus = _make_taus(h, self._n_quantiles)
        return self._compute_quantiles(h, taus)

    @property
    def action_size(self) ->int:
        return self._action_size

    @property
    def encoder(self) ->EncoderWithAction:
        return self._encoder


class ValueFunction(nn.Module):
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), 1)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        h = self._encoder(x)
        return cast(torch.Tensor, self._fc(h))

    def __call__(self, x: torch.Tensor) ->torch.Tensor:
        return cast(torch.Tensor, super().__call__(x))

    def compute_error(self, observations: torch.Tensor, target: torch.Tensor) ->torch.Tensor:
        v_t = self.forward(observations)
        loss = F.mse_loss(v_t, target)
        return loss


class View(nn.Module):
    _shape: Sequence[int]

    def __init__(self, shape: Sequence[int]):
        super().__init__()
        self._shape = shape

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return x.view(self._shape)


class Swish(nn.Module):

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return x * torch.sigmoid(x)


class DummyEncoder(torch.nn.Module):

    def __init__(self, feature_size, action_size=None, concat=False):
        super().__init__()
        self.feature_size = feature_size
        self.observation_shape = feature_size,
        self.action_size = action_size
        self.concat = concat

    def __call__(self, *args):
        if self.concat:
            h = torch.cat([args[0][:, :-args[1].shape[1]], args[1]], dim=1)
            return h
        return args[0]

    def get_feature_size(self):
        return self.feature_size


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VectorEncoder,
     lambda: ([], {'observation_shape': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VectorEncoderWithAction,
     lambda: ([], {'observation_shape': [4, 4], 'action_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (View,
     lambda: ([], {'shape': 4}),
     lambda: ([torch.rand([4])], {}),
     False),
]

class Test_takuseno_d3rlpy(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

