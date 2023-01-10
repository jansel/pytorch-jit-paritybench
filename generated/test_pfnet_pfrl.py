import sys
_module = sys.modules[__name__]
del sys
conf = _module
train_a3c = _module
train_dqn = _module
train_iqn = _module
train_rainbow = _module
train_a2c_ale = _module
train_acer_ale = _module
train_categorical_dqn_ale = _module
train_dqn_ale = _module
train_dqn_batch_ale = _module
train_drqn_ale = _module
train_ppo_ale = _module
train_soft_actor_critic_atlas = _module
train_dqn_batch_grasping = _module
train_categorical_dqn_gym = _module
train_dqn_gym = _module
train_reinforce_gym = _module
train_ddpg = _module
train_ppo = _module
train_soft_actor_critic = _module
train_td3 = _module
train_trpo = _module
optuna_dqn_obs1d = _module
train_rainbow = _module
pfrl = _module
action_value = _module
agent = _module
agents = _module
a2c = _module
a3c = _module
acer = _module
al = _module
categorical_double_dqn = _module
categorical_dqn = _module
ddpg = _module
double_dqn = _module
double_pal = _module
dpp = _module
dqn = _module
iqn = _module
pal = _module
ppo = _module
reinforce = _module
soft_actor_critic = _module
state_q_function_actor = _module
td3 = _module
trpo = _module
collections = _module
persistent_collections = _module
prioritized = _module
random_access_queue = _module
distributions = _module
delta = _module
env = _module
envs = _module
abc = _module
multiprocess_vector_env = _module
serial_vector_env = _module
experiments = _module
evaluation_hooks = _module
evaluator = _module
hooks = _module
prepare_output_dir = _module
train_agent = _module
train_agent_async = _module
train_agent_batch = _module
explorer = _module
explorers = _module
additive_gaussian = _module
additive_ou = _module
boltzmann = _module
epsilon_greedy = _module
greedy = _module
functions = _module
bound_by_tanh = _module
lower_triangular_matrix = _module
initializers = _module
chainer_default = _module
lecun_normal = _module
nn = _module
atari_cnn = _module
branched = _module
concat_obs_and_action = _module
empirical_normalization = _module
lmbda = _module
mlp = _module
mlp_bn = _module
noisy_chain = _module
noisy_linear = _module
recurrent = _module
recurrent_branched = _module
recurrent_sequential = _module
optimizers = _module
rmsprop_eps_inside_sqrt = _module
policies = _module
deterministic_policy = _module
gaussian_policy = _module
softmax_policy = _module
policy = _module
q_function = _module
q_functions = _module
dueling_dqn = _module
state_action_q_functions = _module
state_q_functions = _module
replay_buffer = _module
replay_buffers = _module
episodic = _module
persistent = _module
prioritized_episodic = _module
testing = _module
utils = _module
ask_yes_no = _module
async_ = _module
batch_states = _module
clip_l2_grad_norm = _module
conjugate_gradient = _module
contexts = _module
copy_param = _module
env_modifiers = _module
is_return_code_zero = _module
mode_of_distribution = _module
pretrained_models = _module
random = _module
random_seed = _module
recurrent = _module
reward_filter = _module
stoppable_thread = _module
wrappers = _module
atari_wrappers = _module
cast_observation = _module
continuing_time_limit = _module
monitor = _module
normalize_action_space = _module
randomize_action = _module
render = _module
scale_reward = _module
vector_frame_stack = _module
setup = _module
basetest_ddpg = _module
basetest_dqn_like = _module
basetest_training = _module
test_a2c = _module
test_a3c = _module
test_acer = _module
test_al = _module
test_categorical_dqn = _module
test_ddpg = _module
test_double_categorical_dqn = _module
test_double_dqn = _module
test_double_pal = _module
test_dpp = _module
test_dqn = _module
test_iqn = _module
test_pal = _module
test_ppo = _module
test_reinforce = _module
test_soft_actor_critic = _module
test_td3 = _module
test_trpo = _module
test_persistent_collections = _module
test_prioritized = _module
test_random_access_queue = _module
test_vector_envs = _module
test_evaluation_hooks = _module
test_evaluator = _module
test_hooks = _module
test_prepare_output_dir = _module
test_train_agent = _module
test_train_agent_async = _module
test_train_agent_batch = _module
test_additive_gaussian = _module
test_additive_ou = _module
test_boltzmann = _module
test_epsilon_greedy = _module
test_lower_triangular_matrix = _module
tests_persistent_collections = _module
test_branched = _module
test_empirical_normalization = _module
test_lmbda = _module
test_mlp_bn = _module
test_noisy_chain = _module
test_noisy_linear = _module
test_recurrent_branched = _module
test_recurrent_sequential = _module
basetest_state_action_q_function = _module
test_state_action_q_function = _module
test_persistent_replay_buffer = _module
test_replay_buffer = _module
test_action_value = _module
test_agent = _module
test_testing = _module
test_async = _module
test_batch_states = _module
test_clip_l2_grad_norm = _module
test_conjugate_gradient = _module
test_contexts = _module
test_copy_param = _module
test_is_return_code_zero = _module
test_mode_of_distribution = _module
test_pretrained_models = _module
test_random = _module
test_random_seed = _module
test_recurrent = _module
test_stoppable_thread = _module
test_atari_wrappers = _module
test_cast_observation = _module
test_continuing_time_limit = _module
test_monitor = _module
test_randomize_action = _module
test_render = _module
test_scale_reward = _module
test_vector_frame_stack = _module
plot_scores = _module

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


import numpy as np


from torch import nn


import torch.nn as nn


import torch


import functools


import logging


import torch.optim as optim


from torch import distributions


import random


import warnings


from abc import ABCMeta


from abc import abstractmethod


from abc import abstractproperty


import torch.nn.functional as F


from torch.distributions.utils import lazy_property


from typing import Any


from typing import List


from typing import Optional


from typing import Sequence


from typing import Tuple


from logging import getLogger


import copy


import collections


from torch.nn import functional as F


import time


from logging import Logger


from typing import Callable


from typing import Dict


import itertools


from numbers import Number


from torch.distributions import Distribution


from torch.distributions import constraints


import torch.multiprocessing as mp


from torch.utils.data._utils.collate import default_collate


from torch import distributions as dists


import numpy


import math


class SingleSharedBias(nn.Module):
    """Single shared bias used in the Double DQN paper.

    You can add this link after a Linear layer with nobias=True to implement a
    Linear layer with a single shared bias parameter.

    See http://arxiv.org/abs/1509.06461.
    """

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([1], dtype=torch.float32))

    def __call__(self, x):
        return x + self.bias.expand_as(x)


class ActionValue(object, metaclass=ABCMeta):
    """Struct that holds state-fixed Q-functions and its subproducts.

    Every operation it supports is done in a batch manner.
    """

    @abstractproperty
    def greedy_actions(self):
        """Get argmax_a Q(s,a)."""
        raise NotImplementedError()

    @abstractproperty
    def max(self):
        """Evaluate max Q(s,a)."""
        raise NotImplementedError()

    @abstractmethod
    def evaluate_actions(self, actions):
        """Evaluate Q(s,a) with a = given actions."""
        raise NotImplementedError()

    @abstractproperty
    def params(self):
        """Learnable parameters of this action value.

        Returns:
            tuple of torch.Tensor
        """
        raise NotImplementedError()

    def __getitem__(self, i) ->'ActionValue':
        """ActionValue is expected to be indexable."""
        raise NotImplementedError()


class DiscreteActionValue(ActionValue):
    """Q-function output for discrete action space.

    Args:
        q_values (torch.Tensor):
            Array of Q values whose shape is (batchsize, n_actions)
    """

    def __init__(self, q_values, q_values_formatter=lambda x: x):
        assert isinstance(q_values, torch.Tensor)
        self.device = q_values.device
        self.q_values = q_values
        self.n_actions = q_values.shape[1]
        self.q_values_formatter = q_values_formatter

    @lazy_property
    def greedy_actions(self):
        return self.q_values.detach().argmax(axis=1).int()

    @lazy_property
    def max(self):
        index = self.greedy_actions.long().unsqueeze(1)
        return self.q_values.gather(dim=1, index=index).flatten()

    def evaluate_actions(self, actions):
        index = actions.long().unsqueeze(1)
        return self.q_values.gather(dim=1, index=index).flatten()

    def compute_advantage(self, actions):
        return self.evaluate_actions(actions) - self.max

    def compute_double_advantage(self, actions, argmax_actions):
        return self.evaluate_actions(actions) - self.evaluate_actions(argmax_actions)

    def compute_expectation(self, beta):
        return torch.sum(F.softmax(beta * self.q_values) * self.q_values, dim=1)

    def __repr__(self):
        return 'DiscreteActionValue greedy_actions:{} q_values:{}'.format(self.greedy_actions.detach().cpu().numpy(), self.q_values_formatter(self.q_values.detach().cpu().numpy()))

    @property
    def params(self):
        return self.q_values,

    def __getitem__(self, i):
        return DiscreteActionValue(self.q_values[i], q_values_formatter=self.q_values_formatter)


class DiscreteActionValueHead(nn.Module):

    def forward(self, q_values):
        return DiscreteActionValue(q_values)


class GraspingQFunction(nn.Module):
    """Q-function model for the grasping env.

    This model takes an 84x84 2D image and an integer that indicates the
    number of elapsed steps in an episode as input and outputs action values.
    """

    def __init__(self, n_actions, max_episode_steps):
        super().__init__()
        self.embed = nn.Embedding(max_episode_steps + 1, 3136)
        self.image2hidden = nn.Sequential(nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(), nn.Conv2d(64, 64, 3, stride=1), nn.Flatten())
        self.hidden2out = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, n_actions), DiscreteActionValueHead())

    def forward(self, x):
        image, steps = x
        h = self.image2hidden(image) * torch.sigmoid(self.embed(steps))
        return self.hidden2out(h)


class DistributionalDuelingHead(nn.Module):
    """Head module for defining a distributional dueling network.

    This module expects a (batch_size, in_size)-shaped `torch.Tensor` as input
    and returns `pfrl.action_value.DistributionalDiscreteActionValue`.

    Args:
        in_size (int): Input size.
        n_actions (int): Number of actions.
        n_atoms (int): Number of atoms.
        v_min (float): Minimum value represented by atoms.
        v_max (float): Maximum value represented by atoms.
    """

    def __init__(self, in_size, n_actions, n_atoms, v_min, v_max):
        super().__init__()
        assert in_size % 2 == 0
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.register_buffer('z_values', torch.linspace(v_min, v_max, n_atoms, dtype=torch.float))
        self.a_stream = nn.Linear(in_size // 2, n_actions * n_atoms)
        self.v_stream = nn.Linear(in_size // 2, n_atoms)

    def forward(self, h):
        h_a, h_v = torch.chunk(h, 2, dim=1)
        a_logits = self.a_stream(h_a).reshape((-1, self.n_actions, self.n_atoms))
        a_logits = a_logits - a_logits.mean(dim=1, keepdim=True)
        v_logits = self.v_stream(h_v).reshape((-1, 1, self.n_atoms))
        probs = nn.functional.softmax(a_logits + v_logits, dim=2)
        return pfrl.action_value.DistributionalDiscreteActionValue(probs, self.z_values)


class ACERDiscreteActionHead(nn.Module):
    """ACER model that consists of a separate policy and V-function.

    Args:
        pi (Policy): Policy.
        q (QFunction): Q-function.
    """

    def __init__(self, pi, q):
        super().__init__()
        self.pi = pi
        self.q = q

    def forward(self, obs):
        action_distrib = self.pi(obs)
        action_value = self.q(obs)
        v = (action_distrib.probs * action_value.q_values).sum(1)
        return action_distrib, action_value, v


class SingleActionValue(ActionValue):
    """ActionValue that can evaluate only a single action."""

    def __init__(self, evaluator, maximizer=None):
        self.evaluator = evaluator
        self.maximizer = maximizer

    @lazy_property
    def greedy_actions(self):
        return self.maximizer()

    @lazy_property
    def max(self):
        return self.evaluator(self.greedy_actions)

    def evaluate_actions(self, actions):
        return self.evaluator(actions)

    def compute_advantage(self, actions):
        return self.evaluator(actions) - self.max

    def compute_double_advantage(self, actions, argmax_actions):
        return self.evaluate_actions(actions) - self.evaluate_actions(argmax_actions)

    def __repr__(self):
        return 'SingleActionValue'

    @property
    def params(self):
        warnings.warn('SingleActionValue has no learnable parameters until it is evaluated on some action. If you want to draw a computation graph that outputs SingleActionValue, use the variable returned by its method such as evaluate_actions instead.')
        return ()

    def __getitem__(self, i):
        raise NotImplementedError


class ACERContinuousActionHead(nn.Module):
    """ACER model that consists of a separate policy and V-function.

    Args:
        pi (Policy): Policy.
        v (torch.nn.Module): V-function, a callable mapping from a batch of
            observations to a (batch_size, 1)-shaped `torch.Tensor`.
        adv (StateActionQFunction): Advantage function.
        n (int): Number of samples used to evaluate Q-values.
    """

    def __init__(self, pi, v, adv, n=5):
        super().__init__()
        self.pi = pi
        self.v = v
        self.adv = adv
        self.n = n

    def forward(self, obs):
        action_distrib = self.pi(obs)
        v = self.v(obs)

        def evaluator(action):
            adv_mean = sum(self.adv((obs, action_distrib.sample())) for _ in range(self.n)) / self.n
            return v + self.adv((obs, action)) - adv_mean
        action_value = SingleActionValue(evaluator)
        return action_distrib, action_value, v


def cosine_basis_functions(x, n_basis_functions=64):
    """Cosine basis functions used to embed quantile thresholds.

    Args:
        x (torch.Tensor): Input.
        n_basis_functions (int): Number of cosine basis functions.

    Returns:
        ndarray: Embedding with shape of (x.shape + (n_basis_functions,)).
    """
    i_pi = torch.arange(1, n_basis_functions + 1, dtype=torch.float, device=x.device) * np.pi
    embedding = torch.cos(x[..., None] * i_pi)
    assert embedding.shape == x.shape + (n_basis_functions,)
    return embedding


class CosineBasisLinear(nn.Module):
    """Linear layer following cosine basis functions.

    Args:
        n_basis_functions (int): Number of cosine basis functions.
        out_size (int): Output size.
    """

    def __init__(self, n_basis_functions, out_size):
        super().__init__()
        self.linear = nn.Linear(n_basis_functions, out_size)
        self.n_basis_functions = n_basis_functions
        self.out_size = out_size

    def forward(self, x):
        """Evaluate.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output with shape of (x.shape + (out_size,)).
        """
        h = cosine_basis_functions(x, self.n_basis_functions)
        h = h.reshape(-1, self.n_basis_functions)
        out = self.linear(h)
        out = out.reshape(*x.shape, self.out_size)
        return out


class QuantileDiscreteActionValue(DiscreteActionValue):
    """Quantile action value for discrete actions.
    Args:
        quantiles (torch.Tensor): (batch_size, n_taus, n_actions)
        q_values_formatter (callable):
    """

    def __init__(self, quantiles, q_values_formatter=lambda x: x):
        assert quantiles.ndim == 3
        self.quantiles = quantiles
        self.n_actions = quantiles.shape[2]
        self.q_values_formatter = q_values_formatter

    @lazy_property
    def q_values(self):
        return self.quantiles.mean(1)

    def evaluate_actions_as_quantiles(self, actions):
        """Return the return quantiles of given actions.
        Args:
            actions (torch.Tensor or ndarray): Array of action indices.
                Its shape must be (batch_size,).
        Returns:
            torch.Tensor: Return quantiles. Its shape will be
                (batch_size, n_taus).
        """
        return self.quantiles[torch.arange(self.quantiles.shape[0], dtype=torch.long), :, actions.long()]

    def __repr__(self):
        return 'QuantileDiscreteActionValue greedy_actions:{} q_values:{}'.format(self.greedy_actions.detach().cpu().numpy(), self.q_values_formatter(self.q_values.detach().cpu().numpy()))

    @property
    def params(self):
        return self.quantiles,

    def __getitem__(self, i):
        return QuantileDiscreteActionValue(quantiles=self.quantiles[i], q_values_formatter=self.q_values_formatter)


def _evaluate_psi_x_with_quantile_thresholds(psi_x, phi, f, taus):
    assert psi_x.ndim == 2
    batch_size, hidden_size = psi_x.shape
    assert taus.ndim == 2
    assert taus.shape[0] == batch_size
    n_taus = taus.shape[1]
    phi_taus = phi(taus)
    assert phi_taus.ndim == 3
    assert phi_taus.shape == (batch_size, n_taus, hidden_size)
    h = psi_x.unsqueeze(1) * phi_taus
    h = h.reshape(-1, hidden_size)
    assert h.shape == (batch_size * n_taus, hidden_size)
    h = f(h)
    assert h.ndim == 2
    assert h.shape[0] == batch_size * n_taus
    n_actions = h.shape[-1]
    h = h.reshape(batch_size, n_taus, n_actions)
    return QuantileDiscreteActionValue(h)


class ImplicitQuantileQFunction(nn.Module):
    """Implicit quantile network-based Q-function.

    Args:
        psi (torch.nn.Module): Callable module
            (batch_size, obs_size) -> (batch_size, hidden_size).
        phi (torch.nn.Module): Callable module
            (batch_size, n_taus) -> (batch_size, n_taus, hidden_size).
        f (torch.nn.Module): Callable module
            (batch_size * n_taus, hidden_size)
            -> (batch_size * n_taus, n_actions).

    Returns:
        QuantileDiscreteActionValue: Action values.
    """

    def __init__(self, psi, phi, f):
        super().__init__()
        self.psi = psi
        self.phi = phi
        self.f = f

    def forward(self, x):
        """Evaluate given observations.

        Args:
            x (torch.Tensor): Batch of observations.
        Returns:
            callable: (batch_size, taus) -> (batch_size, taus, n_actions)
        """
        batch_size = x.shape[0]
        psi_x = self.psi(x)
        assert psi_x.ndim == 2
        assert psi_x.shape[0] == batch_size

        def evaluate_with_quantile_thresholds(taus):
            return _evaluate_psi_x_with_quantile_thresholds(psi_x, self.phi, self.f, taus)
        return evaluate_with_quantile_thresholds


class Recurrent(object):
    """Recurrent module interface.

    This class defines the interface of a recurrent module PFRL support.

    The interface is similar to that of `torch.nn.LSTM` except that sequential
    data are expected to be packed in `torch.nn.utils.rnn.PackedSequence`.

    To implement a model with recurrent layers, you can either use
    default container classes such as
    `pfrl.nn.RecurrentSequential` and
    `pfrl.nn.RecurrentBranched` or write your module
    extending this class and `torch.nn.Module`.
    """

    def forward(self, packed_input, recurrent_state):
        """Multi-step batch forward computation.

        Args:
            packed_input (object): Input sequences. Tensors must be packed in
                `torch.nn.utils.rnn.PackedSequence`.
            recurrent_state (object or None): Batched recurrent state.
                If set to None, it is initialized.

        Returns:
            object: Output sequences. Tensors will be packed in
                `torch.nn.utils.rnn.PackedSequence`.
            object or None: New batched recurrent state.
        """
        raise NotImplementedError


class RecurrentImplicitQuantileQFunction(Recurrent, nn.Module):
    """Recurrent implicit quantile network-based Q-function.

    Args:
        psi (torch.nn.Module): Module that implements
            `pfrl.nn.Recurrent`.
            (batch_size, obs_size) -> (batch_size, hidden_size).
        phi (torch.nn.Module): Callable module
            (batch_size, n_taus) -> (batch_size, n_taus, hidden_size).
        f (torch.nn.Module): Callable module
            (batch_size * n_taus, hidden_size)
            -> (batch_size * n_taus, n_actions).

    Returns:
        ImplicitQuantileDiscreteActionValue: Action values.
    """

    def __init__(self, psi, phi, f):
        super().__init__()
        self.psi = psi
        self.phi = phi
        self.f = f

    def forward(self, x, recurrent_state):
        """Evaluate given observations.

        Args:
            x (object): Batched sequences of observations.
            recurrent_state (object): Batched recurrent states.

        Returns:
            callable: (batch_size, taus) -> (batch_size, taus, n_actions)
            object: new recurrent states
        """
        psi_x, recurrent_state = self.psi(x, recurrent_state)
        assert isinstance(psi_x, nn.utils.rnn.PackedSequence)
        psi_x = psi_x.data
        assert psi_x.ndim == 2

        def evaluate_with_quantile_thresholds(taus):
            return _evaluate_psi_x_with_quantile_thresholds(psi_x, self.phi, self.f, taus)
        return evaluate_with_quantile_thresholds, recurrent_state


class TemperatureHolder(nn.Module):
    """Module that holds a temperature as a learnable value.

    Args:
        initial_log_temperature (float): Initial value of log(temperature).
    """

    def __init__(self, initial_log_temperature=0):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(initial_log_temperature, dtype=torch.float32))

    def forward(self):
        """Return a temperature as a torch.Tensor."""
        return torch.exp(self.log_temperature)


def constant_bias_initializer(bias=0.0):

    @torch.no_grad()
    def init_bias(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.bias.fill_(bias)
    return init_bias


def init_lecun_normal(tensor, scale=1.0):
    """Initializes the tensor with LeCunNormal."""
    fan_in = torch.nn.init._calculate_correct_fan(tensor, 'fan_in')
    std = scale * np.sqrt(1.0 / fan_in)
    with torch.no_grad():
        return tensor.normal_(0, std)


@torch.no_grad()
def init_chainer_default(layer):
    """Initializes the layer with the chainer default.
    weights with LeCunNormal(scale=1.0) and zeros as biases
    """
    assert isinstance(layer, nn.Module)
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        init_lecun_normal(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    return layer


class LargeAtariCNN(nn.Module):
    """Large CNN module proposed for DQN in Nature, 2015.

    See: https://www.nature.com/articles/nature14236
    """

    def __init__(self, n_input_channels=4, n_output_channels=512, activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(n_input_channels, 32, 8, stride=4), nn.Conv2d(32, 64, 4, stride=2), nn.Conv2d(64, 64, 3, stride=1)])
        self.output = nn.Linear(3136, n_output_channels)
        self.apply(init_chainer_default)
        self.apply(constant_bias_initializer(bias=bias))

    def forward(self, state):
        h = state
        for layer in self.layers:
            h = self.activation(layer(h))
        h_flat = h.view(h.size(0), -1)
        return self.activation(self.output(h_flat))


class SmallAtariCNN(nn.Module):
    """Small CNN module proposed for DQN in NeurIPS DL Workshop, 2013.

    See: https://arxiv.org/abs/1312.5602
    """

    def __init__(self, n_input_channels=4, n_output_channels=256, activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(n_input_channels, 16, 8, stride=4), nn.Conv2d(16, 32, 4, stride=2)])
        self.output = nn.Linear(2592, n_output_channels)
        self.apply(init_chainer_default)
        self.apply(constant_bias_initializer(bias=bias))

    def forward(self, state):
        h = state
        for layer in self.layers:
            h = self.activation(layer(h))
        h_flat = h.view(h.size(0), -1)
        return self.activation(self.output(h_flat))


class Branched(torch.nn.Module):
    """Module that calls forward functions of child modules in parallel.

    When the `forward` method of this module is called, all the
    arguments are forwarded to each child module's `forward` method.

    The returned values from the child modules are returned as a tuple.

    Args:
        *modules: Child modules. Each module should be callable.
    """

    def __init__(self, *modules):
        super().__init__()
        self.child_modules = torch.nn.ModuleList(modules)

    def forward(self, *args, **kwargs):
        """Forward the arguments to the child modules.

        Args:
            *args, **kwargs: Any arguments forwarded to child modules.  Each
                child module should be able to accept the arguments.

        Returns:
            tuple: Tuple of the returned values from the child modules.
        """
        return tuple(mod(*args, **kwargs) for mod in self.child_modules)


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values.

    Args:
        shape (int or tuple of int): Shape of input values except batch axis.
        batch_axis (int): Batch axis.
        eps (float): Small value for stability.
        dtype (dtype): Dtype of input values.
        until (int or None): If this arg is specified, the link learns input
            values until the sum of batch sizes exceeds it.
    """

    def __init__(self, shape, batch_axis=0, eps=0.01, dtype=np.float32, until=None, clip_threshold=None):
        super(EmpiricalNormalization, self).__init__()
        dtype = np.dtype(dtype)
        self.batch_axis = batch_axis
        self.eps = dtype.type(eps)
        self.until = until
        self.clip_threshold = clip_threshold
        self.register_buffer('_mean', torch.tensor(np.expand_dims(np.zeros(shape, dtype=dtype), batch_axis)))
        self.register_buffer('_var', torch.tensor(np.expand_dims(np.ones(shape, dtype=dtype), batch_axis)))
        self.register_buffer('count', torch.tensor(0))
        self._cached_std_inverse = None

    @property
    def mean(self):
        return torch.squeeze(self._mean, self.batch_axis).clone()

    @property
    def std(self):
        return torch.sqrt(torch.squeeze(self._var, self.batch_axis)).clone()

    @property
    def _std_inverse(self):
        if self._cached_std_inverse is None:
            self._cached_std_inverse = (self._var + self.eps) ** -0.5
        return self._cached_std_inverse

    def experience(self, x):
        """Learn input values without computing the output values of them"""
        if self.until is not None and self.count >= self.until:
            return
        count_x = x.shape[self.batch_axis]
        if count_x == 0:
            return
        self.count += count_x
        rate = count_x / self.count.float()
        assert rate > 0
        assert rate <= 1
        var_x, mean_x = torch.var_mean(x, axis=self.batch_axis, keepdims=True, unbiased=False)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._cached_std_inverse = None

    def forward(self, x, update=True):
        """Normalize mean and variance of values based on emprical values.

        Args:
            x (ndarray or Variable): Input values
            update (bool): Flag to learn the input values

        Returns:
            ndarray or Variable: Normalized output values
        """
        if update:
            self.experience(x)
        normalized = (x - self._mean) * self._std_inverse
        if self.clip_threshold is not None:
            normalized = torch.clamp(normalized, -self.clip_threshold, self.clip_threshold)
        return normalized

    def inverse(self, y):
        std = torch.sqrt(self._var + self.eps)
        return y * std + self._mean


class Lambda(nn.Module):
    """Wraps a callable object to make a `torch.nn.Module`.

    This can be used to add callable objects to `torch.nn.Sequential` or
    `pfrl.nn.RecurrentSequential`, which only accept
    `torch.nn.Module`s.

    Args:
        lambd (callable): Callable object.
    """

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class MLP(nn.Module):
    """Multi-Layer Perceptron"""

    def __init__(self, in_size, out_size, hidden_sizes, nonlinearity=F.relu, last_wscale=1):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        super().__init__()
        if hidden_sizes:
            self.hidden_layers = nn.ModuleList()
            self.hidden_layers.append(nn.Linear(in_size, hidden_sizes[0]))
            for hin, hout in zip(hidden_sizes, hidden_sizes[1:]):
                self.hidden_layers.append(nn.Linear(hin, hout))
            self.hidden_layers.apply(init_chainer_default)
            self.output = nn.Linear(hidden_sizes[-1], out_size)
        else:
            self.output = nn.Linear(in_size, out_size)
        init_lecun_normal(self.output.weight, scale=last_wscale)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        h = x
        if self.hidden_sizes:
            for layer in self.hidden_layers:
                h = self.nonlinearity(layer(h))
        return self.output(h)


class LinearBN(nn.Module):
    """Linear layer with BatchNormalization."""

    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.bn = nn.BatchNorm1d(out_size)

    def forward(self, x):
        return self.bn(self.linear(x))


class MLPBN(nn.Module):
    """Multi-Layer Perceptron with Batch Normalization.

    Args:
        in_size (int): Input size.
        out_size (int): Output size.
        hidden_sizes (list of ints): Sizes of hidden channels.
        normalize_input (bool): If set to True, Batch Normalization is applied
            to inputs.
        normalize_output (bool): If set to True, Batch Normalization is applied
            to outputs.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(self, in_size, out_size, hidden_sizes, normalize_input=True, normalize_output=False, nonlinearity=F.relu, last_wscale=1):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.nonlinearity = nonlinearity
        super().__init__()
        if normalize_input:
            self.input_bn = nn.BatchNorm1d(in_size)
        if hidden_sizes:
            self.hidden_layers = nn.ModuleList()
            self.hidden_layers.append(LinearBN(in_size, hidden_sizes[0]))
            for hin, hout in zip(hidden_sizes, hidden_sizes[1:]):
                self.hidden_layers.append(LinearBN(hin, hout))
            self.output = nn.Linear(hidden_sizes[-1], out_size)
        else:
            self.output = nn.Linear(in_size, out_size)
        init_lecun_normal(self.output.weight, scale=last_wscale)
        if normalize_output:
            self.output_bn = nn.BatchNorm1d(out_size)

    def forward(self, x):
        h = x
        if self.normalize_input:
            h = self.input_bn(h)
        if self.hidden_sizes:
            for layer in self.hidden_layers:
                h = self.nonlinearity(layer(h))
        h = self.output(h)
        if self.normalize_output:
            h = self.output_bn(h)
        return h


def init_lecun_uniform(tensor, scale=1.0):
    """Initializes the tensor with LeCunUniform."""
    fan_in = torch.nn.init._calculate_correct_fan(tensor, 'fan_in')
    s = scale * np.sqrt(3.0 / fan_in)
    with torch.no_grad():
        return tensor.uniform_(-s, s)


def init_variance_scaling_constant(tensor, scale=1.0):
    if tensor.ndim == 1:
        s = scale / np.sqrt(tensor.shape[0])
    else:
        fan_in = torch.nn.init._calculate_correct_fan(tensor, 'fan_in')
        s = scale / np.sqrt(fan_in)
    with torch.no_grad():
        return tensor.fill_(s)


class FactorizedNoisyLinear(nn.Module):
    """Linear layer in Factorized Noisy Network

    Args:
        mu_link (nn.Linear): Linear link that computes mean of output.
        sigma_scale (float): The hyperparameter sigma_0 in the original paper.
            Scaling factor of the initial weights of noise-scaling parameters.
    """

    def __init__(self, mu_link, sigma_scale=0.4):
        super(FactorizedNoisyLinear, self).__init__()
        self._kernel = None
        self.out_size = mu_link.out_features
        self.hasbias = mu_link.bias is not None
        in_size = mu_link.weight.shape[1]
        device = mu_link.weight.device
        self.mu = nn.Linear(in_size, self.out_size, bias=self.hasbias)
        init_lecun_uniform(self.mu.weight, scale=1 / np.sqrt(3))
        self.sigma = nn.Linear(in_size, self.out_size, bias=self.hasbias)
        init_variance_scaling_constant(self.sigma.weight, scale=sigma_scale)
        if self.hasbias:
            init_variance_scaling_constant(self.sigma.bias, scale=sigma_scale)
        self.mu
        self.sigma

    def _eps(self, shape, dtype, device):
        r = torch.normal(mean=0.0, std=1.0, size=(shape,), dtype=dtype, device=device)
        return torch.abs(torch.sqrt(torch.abs(r))) * torch.sign(r)

    def forward(self, x):
        dtype = self.sigma.weight.dtype
        out_size, in_size = self.sigma.weight.shape
        eps = self._eps(in_size + out_size, dtype, self.sigma.weight.device)
        eps_x = eps[:in_size]
        eps_y = eps[in_size:]
        W = torch.addcmul(self.mu.weight, self.sigma.weight, torch.ger(eps_y, eps_x))
        if self.hasbias:
            b = torch.addcmul(self.mu.bias, self.sigma.bias, eps_y)
            return F.linear(x, W, b)
        else:
            return F.linear(x, W)


class RecurrentBranched(Recurrent, nn.ModuleList):
    """Recurrent module that bundles parallel branches.

    This is a recurrent analog to `pfrl.nn.Branched`. It bundles
    multiple recurrent modules.

    Args:
        *modules: Child modules. Each module should be recurrent and callable.
    """

    def __init__(self, *modules):
        super().__init__(modules)

    def forward(self, sequences, recurrent_state):
        if recurrent_state is None:
            n = len(self)
            recurrent_state = [None] * n
        child_ys, rs = tuple(zip(*[link(sequences, rs) for link, rs in zip(self, recurrent_state)]))
        return child_ys, rs


def get_packed_sequence_info(packed):
    """Get `batch_sizes` and `sorted_indices` of `PackedSequence`.

    Args:
        packed (object): Packed sequences. If it contains multiple
            `PackedSequence`s, then only one of them are sampled assuming that
            all of them have same `batch_sizes` and `sorted_indices`.

    Returns:
        Tensor: `PackedSequence.batch_sizes`.
        Tensor: `PackedSequence.sorted_indices`.
    """
    if isinstance(packed, torch.nn.utils.rnn.PackedSequence):
        return packed.batch_sizes, packed.sorted_indices
    if isinstance(packed, tuple):
        for y in packed:
            ret = get_packed_sequence_info(y)
            if ret is not None:
                return ret
    return None


def is_recurrent(layer):
    """Return True iff a given layer is recurrent and supported by PFRL.

    Args:
        layer (callable): Any callable object.

    Returns:
        bool: True iff a given layer is recurrent and supported by PFRL.
    """
    return isinstance(layer, (nn.LSTM, nn.RNN, nn.GRU, Recurrent))


def unwrap_packed_sequences_recursive(packed):
    """Unwrap `PackedSequence` class of packed sequences recursively.

    This function extract `torch.Tensor` that
    `torch.nn.utils.rnn.PackedSequence` holds internally. Sequences in the
    internal tensor is ordered with time axis first.

    Unlike `torch.nn.pad_packed_sequence`, this function just returns the
    underlying tensor as it is without padding.

    To wrap the data by `PackedSequence` again, use
    `wrap_packed_sequences_recursive`.

    Args:
        packed (object): Packed sequences.

    Returns:
        object: Unwrapped packed sequences. If `packed` is a `PackedSequence`,
            then the returned value is `PackedSequence.data`, the underlying
            tensor. If `Packed` is a tuple of `PackedSequence`, then the
            returned value is a tuple of the underlying tensors.
    """
    if isinstance(packed, torch.nn.utils.rnn.PackedSequence):
        return packed.data
    if isinstance(packed, tuple):
        return tuple(unwrap_packed_sequences_recursive(x) for x in packed)
    return packed


def wrap_packed_sequences_recursive(unwrapped, batch_sizes, sorted_indices):
    """Wrap packed tensors by `PackedSequence`.

    Args:
        unwrapped (object): Packed but unwrapped tensor(s).
        batch_sizes (Tensor): See `PackedSequence.batch_sizes`.
        sorted_indices (Tensor): See `PackedSequence.sorted_indices`.

    Returns:
        object: Packed sequences. If `unwrapped` is a tensor, then the returned
            value is a `PackedSequence`. If `unwrapped` is a tuple of tensors,
            then the returned value is a tuple of `PackedSequence`s.
    """
    if isinstance(unwrapped, torch.Tensor):
        return torch.nn.utils.rnn.PackedSequence(unwrapped, batch_sizes=batch_sizes, sorted_indices=sorted_indices)
    if isinstance(unwrapped, tuple):
        return tuple(wrap_packed_sequences_recursive(x, batch_sizes, sorted_indices) for x in unwrapped)
    return unwrapped


class RecurrentSequential(Recurrent, nn.Sequential):
    """Sequential model that can contain stateless recurrent modules.

    This is a recurrent analog to `torch.nn.Sequential`. It supports
    the recurrent interface by automatically detecting recurrent
    modules and handles recurrent states properly.

    For non-recurrent layers, this module automatically concatenates
    the input to the layers for efficient computation.

    Args:
        *layers: Callable objects.
    """

    def forward(self, sequences, recurrent_state):
        if recurrent_state is None:
            recurrent_state_queue = [None] * len(self.recurrent_children)
        else:
            assert len(recurrent_state) == len(self.recurrent_children)
            recurrent_state_queue = list(reversed(recurrent_state))
        new_recurrent_state = []
        h = sequences
        batch_sizes, sorted_indices = get_packed_sequence_info(h)
        is_wrapped = True
        for layer in self:
            if is_recurrent(layer):
                if not is_wrapped:
                    h = wrap_packed_sequences_recursive(h, batch_sizes, sorted_indices)
                    is_wrapped = True
                rs = recurrent_state_queue.pop()
                h, rs = layer(h, rs)
                new_recurrent_state.append(rs)
            else:
                if is_wrapped:
                    h = unwrap_packed_sequences_recursive(h)
                    is_wrapped = False
                h = layer(h)
        if not is_wrapped:
            h = wrap_packed_sequences_recursive(h, batch_sizes, sorted_indices)
        assert not recurrent_state_queue
        assert len(new_recurrent_state) == len(self.recurrent_children)
        return h, tuple(new_recurrent_state)

    @property
    def recurrent_children(self):
        """Return recurrent child modules.

        Returns:
            tuple: Child modules that are recurrent.
        """
        return tuple(child for child in self if is_recurrent(child))


class Delta(Distribution):
    """Delta distribution.

    This is used

    Args:
        loc (float or Tensor): location of the distribution.
    """
    arg_constraints = {'loc': constraints.real}
    support = constraints.real
    has_rsample = True

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return torch.zeros_like(self.loc)

    @property
    def variance(self):
        return torch.zeros_like(self.loc)

    def __init__(self, loc, validate_args=None):
        self.loc = loc
        if isinstance(loc, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Delta, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Delta, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        super(Delta, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape).detach()

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return self.loc.expand(shape)

    def log_prob(self, value):
        raise RuntimeError('Not defined')

    def entropy(self):
        raise RuntimeError('Not defined')


class DeterministicHead(nn.Module):
    """Head module for a deterministic policy."""

    def forward(self, loc):
        return torch.distributions.Independent(Delta(loc=loc), 1)


class GaussianHeadWithStateIndependentCovariance(nn.Module):
    """Gaussian head with state-independent learned covariance.

    This link is intended to be attached to a neural network that outputs
    the mean of a Gaussian policy. The only learnable parameter this link has
    determines the variance in a state-independent way.

    State-independent parameterization of the variance of a Gaussian policy
    is often used with PPO and TRPO, e.g., in https://arxiv.org/abs/1709.06560.

    Args:
        action_size (int): Number of dimensions of the action space.
        var_type (str): Type of parameterization of variance. It must be
            'spherical' or 'diagonal'.
        var_func (callable): Callable that computes the variance from the var
            parameter. It should always return positive values.
        var_param_init (float): Initial value the var parameter.
    """

    def __init__(self, action_size, var_type='spherical', var_func=nn.functional.softplus, var_param_init=0):
        super().__init__()
        self.var_func = var_func
        var_size = {'spherical': 1, 'diagonal': action_size}[var_type]
        self.var_param = nn.Parameter(torch.tensor(np.broadcast_to(var_param_init, var_size), dtype=torch.float))

    def forward(self, mean):
        """Return a Gaussian with given mean.

        Args:
            mean (torch.Tensor or ndarray): Mean of Gaussian.

        Returns:
            torch.distributions.Distribution: Gaussian whose mean is the
                mean argument and whose variance is computed from the parameter
                of this link.
        """
        var = self.var_func(self.var_param)
        return torch.distributions.Independent(torch.distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1)


class GaussianHeadWithDiagonalCovariance(nn.Module):
    """Gaussian head with diagonal covariance.

    This module is intended to be attached to a neural network that outputs
    a vector that is twice the size of an action vector. The vector is split
    and interpreted as the mean and diagonal covariance of a Gaussian policy.

    Args:
        var_func (callable): Callable that computes the variance
            from the second input. It should always return positive values.
    """

    def __init__(self, var_func=nn.functional.softplus):
        super().__init__()
        self.var_func = var_func

    def forward(self, mean_and_var):
        """Return a Gaussian with given mean and diagonal covariance.

        Args:
            mean_and_var (torch.Tensor): Vector that is twice the size of an
                action vector.

        Returns:
            torch.distributions.Distribution: Gaussian distribution with given
                mean and diagonal covariance.
        """
        assert mean_and_var.ndim == 2
        mean, pre_var = mean_and_var.chunk(2, dim=1)
        scale = self.var_func(pre_var).sqrt()
        return torch.distributions.Independent(torch.distributions.Normal(loc=mean, scale=scale), 1)


class GaussianHeadWithFixedCovariance(nn.Module):
    """Gaussian head with fixed covariance.

    This module is intended to be attached to a neural network that outputs
    the mean of a Gaussian policy. Its covariance is fixed to a diagonal matrix
    with a given scale.

    Args:
        scale (float): Scale parameter.
    """

    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale

    def forward(self, mean):
        """Return a Gaussian with given mean.

        Args:
            mean (torch.Tensor): Batch of mean vectors.

        Returns:
            torch.distributions.Distribution: Multivariate Gaussian whose mean
                is the mean argument and whose scale is fixed.
        """
        return torch.distributions.Independent(torch.distributions.Normal(loc=mean, scale=self.scale), 1)


class SoftmaxCategoricalHead(nn.Module):

    def forward(self, logits):
        return torch.distributions.Categorical(logits=logits)


class StateQFunction(object, metaclass=ABCMeta):
    """Abstract Q-function with state input."""

    @abstractmethod
    def __call__(self, x):
        """Evaluates Q-function

        Args:
            x (ndarray): state input

        Returns:
            An instance of ActionValue that allows to calculate the Q-values
            for state x and every possible action
        """
        raise NotImplementedError()


class DuelingDQN(nn.Module, StateQFunction):
    """Dueling Q-Network

    See: http://arxiv.org/abs/1511.06581
    """

    def __init__(self, n_actions, n_input_channels=4, activation=F.relu, bias=0.1):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation
        super().__init__()
        self.conv_layers = nn.ModuleList([nn.Conv2d(n_input_channels, 32, 8, stride=4), nn.Conv2d(32, 64, 4, stride=2), nn.Conv2d(64, 64, 3, stride=1)])
        self.a_stream = MLP(3136, n_actions, [512])
        self.v_stream = MLP(3136, 1, [512])
        self.conv_layers.apply(init_chainer_default)
        self.conv_layers.apply(constant_bias_initializer(bias=bias))

    def forward(self, x):
        h = x
        for layer in self.conv_layers:
            h = self.activation(layer(h))
        batch_size = x.shape[0]
        h = h.reshape(batch_size, -1)
        ya = self.a_stream(h)
        mean = torch.reshape(torch.sum(ya, dim=1) / self.n_actions, (batch_size, 1))
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean
        ys = self.v_stream(h)
        ya, ys = torch.broadcast_tensors(ya, ys)
        q = ya + ys
        return action_value.DiscreteActionValue(q)


class DistributionalDuelingDQN(nn.Module, StateQFunction):
    """Distributional dueling fully-connected Q-function with discrete actions."""

    def __init__(self, n_actions, n_atoms, v_min, v_max, n_input_channels=4, activation=torch.relu, bias=0.1):
        assert n_atoms >= 2
        assert v_min < v_max
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_atoms = n_atoms
        super().__init__()
        self.z_values = torch.linspace(v_min, v_max, n_atoms, dtype=torch.float32)
        self.conv_layers = nn.ModuleList([nn.Conv2d(n_input_channels, 32, 8, stride=4), nn.Conv2d(32, 64, 4, stride=2), nn.Conv2d(64, 64, 3, stride=1)])
        self.main_stream = nn.Linear(3136, 1024)
        self.a_stream = nn.Linear(512, n_actions * n_atoms)
        self.v_stream = nn.Linear(512, n_atoms)
        self.apply(init_chainer_default)
        self.conv_layers.apply(constant_bias_initializer(bias=bias))

    def forward(self, x):
        h = x
        for layer in self.conv_layers:
            h = self.activation(layer(h))
        batch_size = x.shape[0]
        h = self.activation(self.main_stream(h.view(batch_size, -1)))
        h_a, h_v = torch.chunk(h, 2, dim=1)
        ya = self.a_stream(h_a).reshape((batch_size, self.n_actions, self.n_atoms))
        mean = ya.sum(dim=1, keepdim=True) / self.n_actions
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean
        ys = self.v_stream(h_v).reshape((batch_size, 1, self.n_atoms))
        ya, ys = torch.broadcast_tensors(ya, ys)
        q = F.softmax(ya + ys, dim=2)
        self.z_values = self.z_values
        return action_value.DistributionalDiscreteActionValue(q, self.z_values)


class StateActionQFunction(object, metaclass=ABCMeta):
    """Abstract Q-function with state and action input."""

    @abstractmethod
    def __call__(self, x, a):
        """Evaluates Q-function

        Args:
            x (ndarray): state input
            a (ndarray): action input

        Returns:
            Q-value for state x and action a
        """
        raise NotImplementedError()


class SingleModelStateActionQFunction(nn.Module, StateActionQFunction):
    """Q-function with discrete actions.

    Args:
        model (nn.Module):
            Module that is callable and outputs action values.
    """

    def __init__(self, model):
        super().__init__(model=model)

    def forward(self, x, a):
        h = self.model(x, a)
        return h


class FCSAQFunction(MLP, StateActionQFunction):
    """Fully-connected (s,a)-input Q-function.

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported. It is not used if n_hidden_layers is zero.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels, n_hidden_layers, nonlinearity=F.relu, last_wscale=1.0):
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.nonlinearity = nonlinearity
        super().__init__(in_size=self.n_input_channels, out_size=1, hidden_sizes=[self.n_hidden_channels] * self.n_hidden_layers, nonlinearity=nonlinearity, last_wscale=last_wscale)

    def forward(self, state, action):
        h = torch.cat((state, action), dim=1)
        return super().forward(h)


class FCLSTMSAQFunction(nn.Module, StateActionQFunction):
    """Fully-connected + LSTM (s,a)-input Q-function.

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels, n_hidden_layers, nonlinearity=F.relu, last_wscale=1.0):
        raise NotImplementedError()
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.nonlinearity = nonlinearity
        super().__init__()
        self.fc = MLP(self.n_input_channels, n_hidden_channels, [self.n_hidden_channels] * self.n_hidden_layers, nonlinearity=nonlinearity)
        self.lstm = nn.LSTM(num_layers=1, input_size=n_hidden_channels, hidden_size=n_hidden_channels)
        self.out = nn.Linear(n_hidden_channels, 1)
        for n, p in self.lstm.named_parameters():
            if 'weight' in n:
                init_lecun_normal(p)
            else:
                nn.init.zeros_(p)
        init_lecun_normal(self.out.weight, scale=last_wscale)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, a):
        h = torch.cat((x, a), dim=1)
        h = self.nonlinearity(self.fc(h))
        h = self.lstm(h)
        return self.out(h)


class FCBNSAQFunction(MLPBN, StateActionQFunction):
    """Fully-connected + BN (s,a)-input Q-function.

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers.
        normalize_input (bool): If set to True, Batch Normalization is applied
            to both observations and actions.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported. It is not used if n_hidden_layers is zero.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels, n_hidden_layers, normalize_input=True, nonlinearity=F.relu, last_wscale=1.0):
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.normalize_input = normalize_input
        self.nonlinearity = nonlinearity
        super().__init__(in_size=self.n_input_channels, out_size=1, hidden_sizes=[self.n_hidden_channels] * self.n_hidden_layers, normalize_input=self.normalize_input, nonlinearity=nonlinearity, last_wscale=last_wscale)

    def forward(self, state, action):
        h = torch.cat((state, action), dim=1)
        return super().forward(h)


class FCBNLateActionSAQFunction(nn.Module, StateActionQFunction):
    """Fully-connected + BN (s,a)-input Q-function with late action input.

    Actions are not included until the second hidden layer and not normalized.
    This architecture is used in the DDPG paper:
    http://arxiv.org/abs/1509.02971

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers. It must be greater than
            or equal to 1.
        normalize_input (bool): If set to True, Batch Normalization is applied
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels, n_hidden_layers, normalize_input=True, nonlinearity=F.relu, last_wscale=1.0):
        assert n_hidden_layers >= 1
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.normalize_input = normalize_input
        self.nonlinearity = nonlinearity
        super().__init__()
        self.obs_mlp = MLPBN(in_size=n_dim_obs, out_size=n_hidden_channels, hidden_sizes=[], normalize_input=normalize_input, normalize_output=True)
        self.mlp = MLP(in_size=n_hidden_channels + n_dim_action, out_size=1, hidden_sizes=[self.n_hidden_channels] * (self.n_hidden_layers - 1), nonlinearity=nonlinearity, last_wscale=last_wscale)
        self.output = self.mlp.output

    def forward(self, state, action):
        h = self.nonlinearity(self.obs_mlp(state))
        h = torch.cat((h, action), dim=1)
        return self.mlp(h)


class FCLateActionSAQFunction(nn.Module, StateActionQFunction):
    """Fully-connected (s,a)-input Q-function with late action input.

    Actions are not included until the second hidden layer and not normalized.
    This architecture is used in the DDPG paper:
    http://arxiv.org/abs/1509.02971

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers. It must be greater than
            or equal to 1.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels, n_hidden_layers, nonlinearity=F.relu, last_wscale=1.0):
        assert n_hidden_layers >= 1
        self.n_input_channels = n_dim_obs + n_dim_action
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.nonlinearity = nonlinearity
        super().__init__()
        self.obs_mlp = MLP(in_size=n_dim_obs, out_size=n_hidden_channels, hidden_sizes=[])
        self.mlp = MLP(in_size=n_hidden_channels + n_dim_action, out_size=1, hidden_sizes=[self.n_hidden_channels] * (self.n_hidden_layers - 1), nonlinearity=nonlinearity, last_wscale=last_wscale)
        self.output = self.mlp.output

    def forward(self, state, action):
        h = self.nonlinearity(self.obs_mlp(state))
        h = torch.cat((h, action), dim=1)
        return self.mlp(h)


class SingleModelStateQFunctionWithDiscreteAction(nn.Module, StateQFunction):
    """Q-function with discrete actions.

    Args:
        model (nn.Module):
            Model that is callable and outputs action values.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        h = self.model(x)
        return DiscreteActionValue(h)


class FCStateQFunctionWithDiscreteAction(SingleModelStateQFunctionWithDiscreteAction):
    """Fully-connected state-input Q-function with discrete actions.

    Args:
        n_dim_obs: number of dimensions of observation space
        n_actions (int): Number of actions in action space.
        n_hidden_channels: number of hidden channels
        n_hidden_layers: number of hidden layers
        nonlinearity (callable): Nonlinearity applied after each hidden layer.
        last_wscale (float): Weight scale of the last layer.
    """

    def __init__(self, ndim_obs, n_actions, n_hidden_channels, n_hidden_layers, nonlinearity=F.relu, last_wscale=1.0):
        super().__init__(model=MLP(in_size=ndim_obs, out_size=n_actions, hidden_sizes=[n_hidden_channels] * n_hidden_layers, nonlinearity=nonlinearity, last_wscale=last_wscale))


class DistributionalDiscreteActionValue(ActionValue):
    """distributional Q-function output for discrete action space.

    Args:
        q_dist: Probabilities of atoms. Its shape must be
            (batchsize, n_actions, n_atoms).
        z_values (ndarray): Values represented by atoms.
            Its shape must be (n_atoms,).
    """

    def __init__(self, q_dist, z_values, q_values_formatter=lambda x: x):
        assert isinstance(q_dist, torch.Tensor)
        assert isinstance(z_values, torch.Tensor)
        assert q_dist.ndim == 3
        assert z_values.ndim == 1
        assert q_dist.shape[2] == z_values.shape[0]
        self.z_values = z_values
        q_scaled = q_dist * self.z_values[None, None, ...]
        self.q_values = q_scaled.sum(dim=2)
        self.q_dist = q_dist
        self.n_actions = q_dist.shape[1]
        self.q_values_formatter = q_values_formatter

    @lazy_property
    def greedy_actions(self):
        return self.q_values.argmax(dim=1).detach()

    @lazy_property
    def max(self):
        return torch.gather(self.q_values, 1, self.greedy_actions[:, None])[:, 0]

    @lazy_property
    def max_as_distribution(self):
        """Return the return distributions of the greedy actions.

        Returns:
            torch.Tensor: Return distributions. Its shape will be
                (batch_size, n_atoms).
        """
        return self.q_dist[torch.arange(self.q_values.shape[0]), self.greedy_actions.detach()]

    def evaluate_actions(self, actions):
        return torch.gather(self.q_values, 1, actions[:, None])[:, 0]

    def evaluate_actions_as_distribution(self, actions):
        """Return the return distributions of given actions.

        Args:
            actions (torch.Tensor): Array of action indices.
                Its shape must be (batch_size,).

        Returns:
            torch.Tensor: Return distributions. Its shape will be
                (batch_size, n_atoms).
        """
        return self.q_dist[torch.arange(self.q_values.shape[0]), actions]

    def compute_advantage(self, actions):
        return self.evaluate_actions(actions) - self.max

    def compute_double_advantage(self, actions, argmax_actions):
        return self.evaluate_actions(actions) - self.evaluate_actions(argmax_actions)

    def compute_expectation(self, beta):
        return (F.softmax(beta * self.q_values) * self.q_values).sum(dim=1)

    def __repr__(self):
        return 'DistributionalDiscreteActionValue greedy_actions:{} q_values:{}'.format(self.greedy_actions.detach(), self.q_values_formatter(self.q_values.detach()))

    @property
    def params(self):
        return self.q_dist,

    def __getitem__(self, i):
        return DistributionalDiscreteActionValue(self.q_dist[i], self.z_values, q_values_formatter=self.q_values_formatter)


class DistributionalSingleModelStateQFunctionWithDiscreteAction(nn.Module, StateQFunction):
    """Distributional Q-function with discrete actions.

    Args:
        model (nn.Module):
            model that is callable and outputs atoms for each action.
        z_values (ndarray): Returns represented by atoms. Its shape must be
            (n_atoms,).
    """

    def __init__(self, model, z_values):
        super().__init__()
        self.model = model
        self.register_buffer('z_values', torch.from_numpy(z_values))

    def forward(self, x):
        h = self.model(x)
        return DistributionalDiscreteActionValue(h, self.z_values)


class DistributionalFCStateQFunctionWithDiscreteAction(DistributionalSingleModelStateQFunctionWithDiscreteAction):
    """Distributional fully-connected Q-function with discrete actions.

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_actions (int): Number of actions in action space.
        n_atoms (int): Number of atoms of return distribution.
        v_min (float): Minimum value this model can approximate.
        v_max (float): Maximum value this model can approximate.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers.
        nonlinearity (callable): Nonlinearity applied after each hidden layer.
        last_wscale (float): Weight scale of the last layer.
    """

    def __init__(self, ndim_obs, n_actions, n_atoms, v_min, v_max, n_hidden_channels, n_hidden_layers, nonlinearity=F.relu, last_wscale=1.0):
        assert n_atoms >= 2
        assert v_min < v_max
        z_values = np.linspace(v_min, v_max, num=n_atoms, dtype=np.float32)
        model = nn.Sequential(MLP(in_size=ndim_obs, out_size=n_actions * n_atoms, hidden_sizes=[n_hidden_channels] * n_hidden_layers, nonlinearity=nonlinearity, last_wscale=last_wscale), Lambda(lambda x: torch.reshape(x, (-1, n_actions, n_atoms))), nn.Softmax(dim=2))
        super().__init__(model=model, z_values=z_values)


class QuadraticActionValue(ActionValue):
    """Q-function output for continuous action space.

    See: http://arxiv.org/abs/1603.00748

    Define a Q(s,a) with A(s,a) in a quadratic form.

    Q(s,a) = V(s,a) + A(s,a)
    A(s,a) = -1/2 (u - mu(s))^T P(s) (u - mu(s))

    Args:
        mu (torch.Tensor): mu(s), actions that maximize A(s,a)
        mat (torch.Tensor): P(s), coefficient matrices of A(s,a).
          It must be positive definite.
        v (torch.Tensor): V(s), values of s
        min_action (ndarray): minimum action, not batched
        max_action (ndarray): maximum action, not batched
    """

    def __init__(self, mu, mat, v, min_action=None, max_action=None):
        self.mu = mu
        self.mat = mat
        self.v = v
        self.device = mu.device
        if isinstance(min_action, (int, float)):
            min_action = [min_action]
        if min_action is None:
            self.min_action = None
        else:
            self.min_action = torch.as_tensor(min_action).float()
        if isinstance(max_action, (int, float)):
            max_action = [max_action]
        if max_action is None:
            self.max_action = None
        else:
            self.max_action = torch.as_tensor(max_action).float()
        self.batch_size = self.mu.shape[0]

    @lazy_property
    def greedy_actions(self):
        a = self.mu
        if self.min_action is not None:
            a = torch.max(self.min_action.unsqueeze(0).expand_as(a), a)
        if self.max_action is not None:
            a = torch.min(self.max_action.unsqueeze(0).expand_as(a), a)
        return a

    @lazy_property
    def max(self):
        if self.min_action is None and self.max_action is None:
            return self.v.reshape(self.batch_size)
        else:
            return self.evaluate_actions(self.greedy_actions)

    def evaluate_actions(self, actions):
        u_minus_mu = actions - self.mu
        a = -0.5 * torch.matmul(torch.matmul(u_minus_mu[:, None, :], self.mat), u_minus_mu[:, :, None])[:, 0, 0]
        return a + self.v.reshape(self.batch_size)

    def compute_advantage(self, actions):
        return self.evaluate_actions(actions) - self.max

    def compute_double_advantage(self, actions, argmax_actions):
        return self.evaluate_actions(actions) - self.evaluate_actions(argmax_actions)

    def __repr__(self):
        return 'QuadraticActionValue greedy_actions:{} v:{}'.format(self.greedy_actions.detach().cpu().numpy(), self.v.detach().cpu().numpy())

    @property
    def params(self):
        return self.mu, self.mat, self.v

    def __getitem__(self, i):
        return QuadraticActionValue(self.mu[i], self.mat[i], self.v[i], min_action=self.min_action, max_action=self.max_action)


def set_batch_diagonal(array, diag_val):
    batch_size, m, n = array.shape
    assert m == n
    rows, cols = np.diag_indices(n)
    array[:, rows, cols] = diag_val


def set_batch_non_diagonal(array, non_diag_val):
    batch_size, m, n = array.shape
    assert m == n
    rows, cols = np.tril_indices(n, -1)
    array[:, rows, cols] = non_diag_val


def lower_triangular_matrix(diag, non_diag):
    assert isinstance(diag, torch.Tensor)
    assert isinstance(non_diag, torch.Tensor)
    batch_size = diag.shape[0]
    n = diag.shape[1]
    y = torch.zeros((batch_size, n, n), dtype=torch.float32)
    y = y
    set_batch_non_diagonal(y, non_diag)
    set_batch_diagonal(y, diag)
    return y


def scale_by_tanh(x, low, high):
    scale = (high - low) / 2
    scale = torch.unsqueeze(torch.from_numpy(scale), dim=0)
    mean = (high + low) / 2
    mean = torch.unsqueeze(torch.from_numpy(mean), dim=0)
    return torch.tanh(x) * scale + mean


class FCQuadraticStateQFunction(nn.Module, StateQFunction):
    """Fully-connected state-input continuous Q-function.

    See: https://arxiv.org/abs/1603.00748

    Args:
        n_input_channels: number of input channels
        n_dim_action: number of dimensions of action space
        n_hidden_channels: number of hidden channels
        n_hidden_layers: number of hidden layers
        action_space: action_space
        scale_mu (bool): scale mu by applying tanh if True
    """

    def __init__(self, n_input_channels, n_dim_action, n_hidden_channels, n_hidden_layers, action_space, scale_mu=True):
        self.n_input_channels = n_input_channels
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_channels = n_hidden_channels
        self.n_dim_action = n_dim_action
        assert action_space is not None
        self.scale_mu = scale_mu
        self.action_space = action_space
        super().__init__()
        hidden_layers = nn.ModuleList()
        assert n_hidden_layers >= 1
        hidden_layers.append(init_chainer_default(nn.Linear(n_input_channels, n_hidden_channels)))
        for _ in range(n_hidden_layers - 1):
            hidden_layers.append(init_chainer_default(nn.Linear(n_hidden_channels, n_hidden_channels)))
        self.hidden_layers = hidden_layers
        self.v = init_chainer_default(nn.Linear(n_hidden_channels, 1))
        self.mu = init_chainer_default(nn.Linear(n_hidden_channels, n_dim_action))
        self.mat_diag = init_chainer_default(nn.Linear(n_hidden_channels, n_dim_action))
        non_diag_size = n_dim_action * (n_dim_action - 1) // 2
        if non_diag_size > 0:
            self.mat_non_diag = init_chainer_default(nn.Linear(n_hidden_channels, non_diag_size))

    def forward(self, state):
        h = state
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        v = self.v(h)
        mu = self.mu(h)
        if self.scale_mu:
            mu = scale_by_tanh(mu, high=self.action_space.high, low=self.action_space.low)
        mat_diag = torch.exp(self.mat_diag(h))
        if hasattr(self, 'mat_non_diag'):
            mat_non_diag = self.mat_non_diag(h)
            tril = lower_triangular_matrix(mat_diag, mat_non_diag)
            mat = torch.matmul(tril, torch.transpose(tril, 1, 2))
        else:
            mat = torch.unsqueeze(mat_diag ** 2, dim=2)
        return QuadraticActionValue(mu, mat, v, min_action=self.action_space.low, max_action=self.action_space.high)


class NonCudnnLSTM(nn.LSTM):
    """Non-cuDNN LSTM that supports double backprop.

    This is a workaround to address the issue that cuDNN RNNs in PyTorch
    do not support double backprop.

    See https://github.com/pytorch/pytorch/issues/5261.
    """

    def forward(self, x, recurrent_state):
        with torch.backends.cudnn.flags(enabled=False):
            return super().forward(x, recurrent_state)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ACERContinuousActionHead,
     lambda: ([], {'pi': _mock_layer(), 'v': _mock_layer(), 'adv': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Branched,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (CosineBasisLinear,
     lambda: ([], {'n_basis_functions': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DeterministicHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DiscreteActionValueHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DistributionalDuelingHead,
     lambda: ([], {'in_size': 4, 'n_actions': 4, 'n_atoms': 4, 'v_min': 4, 'v_max': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (EmpiricalNormalization,
     lambda: ([], {'shape': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (FCBNLateActionSAQFunction,
     lambda: ([], {'n_dim_obs': 4, 'n_dim_action': 4, 'n_hidden_channels': 4, 'n_hidden_layers': 1}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (FCBNSAQFunction,
     lambda: ([], {'n_dim_obs': 4, 'n_dim_action': 4, 'n_hidden_channels': 4, 'n_hidden_layers': 1}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (FCLateActionSAQFunction,
     lambda: ([], {'n_dim_obs': 4, 'n_dim_action': 4, 'n_hidden_channels': 4, 'n_hidden_layers': 1}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (FCSAQFunction,
     lambda: ([], {'n_dim_obs': 4, 'n_dim_action': 4, 'n_hidden_channels': 4, 'n_hidden_layers': 1}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (FCStateQFunctionWithDiscreteAction,
     lambda: ([], {'ndim_obs': 4, 'n_actions': 4, 'n_hidden_channels': 4, 'n_hidden_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GaussianHeadWithDiagonalCovariance,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (GaussianHeadWithFixedCovariance,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GaussianHeadWithStateIndependentCovariance,
     lambda: ([], {'action_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ImplicitQuantileQFunction,
     lambda: ([], {'psi': _mock_layer(), 'phi': 4, 'f': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Lambda,
     lambda: ([], {'lambd': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearBN,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (SingleModelStateQFunctionWithDiscreteAction,
     lambda: ([], {'model': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SingleSharedBias,
     lambda: ([], {}),
     lambda: ([], {'x': torch.rand([4, 4])}),
     True),
    (SoftmaxCategoricalHead,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TemperatureHolder,
     lambda: ([], {}),
     lambda: ([], {}),
     True),
]

class Test_pfnet_pfrl(_paritybench_base):
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

