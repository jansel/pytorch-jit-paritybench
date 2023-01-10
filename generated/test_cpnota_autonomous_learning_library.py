import sys
_module = sys.modules[__name__]
del sys
all = _module
agents = _module
_agent = _module
_multiagent = _module
_parallel_agent = _module
a2c = _module
c51 = _module
ddpg = _module
ddqn = _module
dqn = _module
independent = _module
ppo = _module
rainbow = _module
sac = _module
vac = _module
vpg = _module
vqn = _module
vsarsa = _module
approximation = _module
approximation = _module
checkpointer = _module
feature_network = _module
feature_network_test = _module
identity = _module
identity_test = _module
q_continuous = _module
q_dist = _module
q_dist_test = _module
q_network = _module
q_network_test = _module
target = _module
abstract = _module
fixed = _module
polyak = _module
trivial = _module
v_network = _module
v_network_test = _module
bodies = _module
_body = _module
atari = _module
rewards = _module
time = _module
time_test = _module
vision = _module
core = _module
state = _module
state_test = _module
environments = _module
_environment = _module
_multiagent_environment = _module
_vector_environment = _module
atari = _module
atari_test = _module
atari_wrappers = _module
duplicate_env = _module
duplicate_env_test = _module
gym = _module
gym_test = _module
multiagent_atari = _module
multiagent_atari_test = _module
multiagent_pettingzoo = _module
multiagent_pettingzoo_test = _module
pybullet = _module
pybullet_test = _module
vector_env = _module
vector_env_test = _module
experiments = _module
experiment = _module
multiagent_env_experiment = _module
multiagent_env_experiment_test = _module
parallel_env_experiment = _module
parallel_env_experiment_test = _module
plots = _module
run_experiment = _module
single_env_experiment = _module
single_env_experiment_test = _module
slurm = _module
watch = _module
logging = _module
_logger = _module
dummy = _module
experiment = _module
memory = _module
advantage = _module
advantage_test = _module
generalized_advantage = _module
generalized_advantage_test = _module
replay_buffer = _module
replay_buffer_test = _module
segment_tree = _module
nn = _module
nn_test = _module
optim = _module
scheduler = _module
scheduler_test = _module
policies = _module
deterministic = _module
deterministic_test = _module
gaussian = _module
gaussian_test = _module
greedy = _module
soft_deterministic = _module
soft_deterministic_test = _module
softmax = _module
softmax_test = _module
presets = _module
a2c = _module
c51 = _module
ddqn = _module
dqn = _module
models = _module
test_ = _module
ppo = _module
rainbow = _module
vac = _module
vpg = _module
vqn = _module
vsarsa = _module
atari_test = _module
builder = _module
classic_control = _module
a2c = _module
c51 = _module
ddqn = _module
dqn = _module
ppo = _module
rainbow = _module
vac = _module
vpg = _module
vqn = _module
vsarsa = _module
classic_control_test = _module
continuous = _module
ddpg = _module
models = _module
ppo = _module
sac = _module
continuous_test = _module
independent_multiagent = _module
multiagent_atari_test = _module
preset = _module
atari40 = _module
conf = _module
examples = _module
slurm_experiment = _module
atari_test = _module
multiagent_atari_test = _module
validate_agent = _module
scripts = _module
classic = _module
plot = _module
release = _module
watch_atari = _module
watch_classic = _module
watch_continuous = _module
watch_multiagent_atari = _module
setup = _module

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


from abc import ABC


from abc import abstractmethod


import torch


from torch.nn.functional import mse_loss


import numpy as np


from torch.distributions.normal import Normal


from torch.nn import utils


import warnings


from torch import nn


from torch.nn import functional as F


from torch.nn.functional import smooth_l1_loss


import copy


import time


from torch.utils.tensorboard import SummaryWriter


import random


from torch.nn import *


from torch.distributions.independent import Independent


from torch.nn import functional


import math


from torch.optim import Adam


from torch.optim.lr_scheduler import CosineAnnealingLR


class FeatureModule(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, states):
        features = states.as_output(self.model(states.as_input('observation')))
        return states.update('observation', features)


class QDistModule(torch.nn.Module):

    def __init__(self, model, n_actions, atoms):
        super().__init__()
        self.atoms = atoms
        self.n_actions = n_actions
        self.n_atoms = len(atoms)
        self.device = next(model.parameters()).device
        self.terminal = torch.zeros(self.n_atoms)
        self.terminal[self.n_atoms // 2] = 1.0
        self.model = nn.RLNetwork(model)
        self.count = 0

    def forward(self, states, actions=None):
        values = self.model(states).view((len(states), self.n_actions, self.n_atoms))
        values = F.softmax(values, dim=2)
        mask = states.mask
        if torch.is_tensor(mask):
            values = (values - self.terminal) * states.mask.view((-1, 1, 1)).float() + self.terminal
        else:
            values = (values - self.terminal) * mask + self.terminal
        if actions is None:
            return values
        if isinstance(actions, list):
            actions = torch.cat(actions)
        return values[torch.arange(len(states)), actions]

    def to(self, device):
        self.device = device
        self.atoms = self.atoms
        self.terminal = self.terminal
        return super()


class RLNetwork(nn.Module):
    """
    Wraps a network such that States can be given as input.
    """

    def __init__(self, model, _=None):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device

    def forward(self, state):
        return state.apply(self.model, 'observation')


class Aggregation(nn.Module):
    """
    Aggregation layer for the Dueling architecture.

    https://arxiv.org/abs/1511.06581
    This layer computes a Q function by combining
    an estimate of V with an estimate of the advantage.
    The advantage is normalized by subtracting the average
    advantage so that we can properly
    """

    def forward(self, value, advantages):
        return value + advantages - torch.mean(advantages, dim=1, keepdim=True)


class Dueling(nn.Module):
    """
    Implementation of the head for the Dueling architecture.

    https://arxiv.org/abs/1511.06581
    This module computes a Q function by computing
    an estimate of V, and estimate of the advantage,
    and combining them with a special Aggregation layer.
    """

    def __init__(self, value_model, advantage_model):
        super().__init__()
        self.value_model = value_model
        self.advantage_model = advantage_model
        self.aggregation = Aggregation()

    def forward(self, features):
        value = self.value_model(features)
        advantages = self.advantage_model(features)
        return self.aggregation(value, advantages)


class CategoricalDueling(nn.Module):
    """Dueling architecture for C51/Rainbow"""

    def __init__(self, value_model, advantage_model):
        super().__init__()
        self.value_model = value_model
        self.advantage_model = advantage_model

    def forward(self, features):
        batch_size = len(features)
        value_dist = self.value_model(features)
        atoms = value_dist.shape[1]
        advantage_dist = self.advantage_model(features).view((batch_size, -1, atoms))
        advantage_mean = advantage_dist.mean(dim=1, keepdim=True)
        return (value_dist.view((batch_size, 1, atoms)) + advantage_dist - advantage_mean).view((batch_size, -1))


class Flatten(nn.Module):
    """
    Flatten a tensor, e.g., between conv2d and linear layers.

    The maintainers FINALLY added this to torch.nn, but I am
    leaving it in for compatible for the moment.
    """

    def forward(self, x):
        return x.view(x.size()[0], -1)


class NoisyLinear(nn.Linear):
    """
    Implementation of Linear layer for NoisyNets

    https://arxiv.org/abs/1706.10295
    NoisyNets are a replacement for epsilon greedy exploration.
    Gaussian noise is added to the weights of the output layer, resulting in
    a stochastic policy. Exploration is implicitly learned at a per-state
    and per-action level, resulting in smarter exploration.
    """

    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))
            self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = np.sqrt(3 / self.in_features)
        nn.init.uniform_(self.weight, -std, std)
        nn.init.uniform_(self.bias, -std, std)

    def forward(self, x):
        bias = self.bias
        if not self.training:
            return F.linear(x, self.weight, bias)
        torch.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
        if self.bias is not None:
            torch.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(x, self.weight + self.sigma_weight * self.epsilon_weight, bias)


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise

    N.B. nn.Linear already initializes weight and bias to
    """

    def __init__(self, in_features, out_features, sigma_init=0.4, init_scale=3, bias=True):
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_init / np.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer('epsilon_input', torch.zeros(1, in_features))
        self.register_buffer('epsilon_output', torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))

    def reset_parameters(self):
        std = np.sqrt(self.init_scale / self.in_features)
        nn.init.uniform_(self.weight, -std, std)
        nn.init.uniform_(self.bias, -std, std)

    def forward(self, input):
        if not self.training:
            return F.linear(input, self.weight, self.bias)
        torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
        torch.randn(self.epsilon_output.size(), out=self.epsilon_output)
        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input)
        eps_out = func(self.epsilon_output)
        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)


class Linear0(nn.Linear):

    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.0)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class Scale(nn.Module):

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class TanhActionBound(nn.Module):

    def __init__(self, action_space):
        super().__init__()
        self.register_buffer('weight', torch.tensor((action_space.high - action_space.low) / 2))
        self.register_buffer('bias', torch.tensor((action_space.high + action_space.low) / 2))

    def forward(self, x):
        return torch.tanh(x) * self.weight + self.bias


class DeterministicPolicyNetwork(RLNetwork):

    def __init__(self, model, space):
        super().__init__(model)
        self._action_dim = space.shape[0]
        self._tanh_scale = torch.tensor((space.high - space.low) / 2)
        self._tanh_mean = torch.tensor((space.high + space.low) / 2)

    def forward(self, state):
        return self._squash(super().forward(state))

    def _squash(self, x):
        return torch.tanh(x) * self._tanh_scale + self._tanh_mean

    def to(self, device):
        self._tanh_mean = self._tanh_mean
        self._tanh_scale = self._tanh_scale
        return super()


class GaussianPolicyNetwork(RLNetwork):

    def __init__(self, model, space):
        super().__init__(model)
        self._center = torch.tensor((space.high + space.low) / 2)
        self._scale = torch.tensor((space.high - space.low) / 2)

    def forward(self, state):
        outputs = super().forward(state)
        action_dim = outputs.shape[-1] // 2
        means = outputs[..., 0:action_dim]
        logvars = outputs[..., action_dim:]
        std = (0.5 * logvars).exp_()
        return Independent(Normal(means + self._center, std * self._scale), 1)

    def to(self, device):
        self._center = self._center
        self._scale = self._scale
        return super()


class SoftDeterministicPolicyNetwork(RLNetwork):

    def __init__(self, model, space):
        super().__init__(model)
        self._action_dim = space.shape[0]
        self._tanh_scale = torch.tensor((space.high - space.low) / 2)
        self._tanh_mean = torch.tensor((space.high + space.low) / 2)

    def forward(self, state):
        outputs = super().forward(state)
        normal = self._normal(outputs)
        if self.training:
            action, log_prob = self._sample(normal)
            return action, log_prob
        return self._squash(normal.loc)

    def _normal(self, outputs):
        means = outputs[..., 0:self._action_dim]
        logvars = outputs[..., self._action_dim:]
        std = logvars.mul(0.5).exp_()
        return torch.distributions.normal.Normal(means, std)

    def _sample(self, normal):
        raw = normal.rsample()
        log_prob = self._log_prob(normal, raw)
        return self._squash(raw), log_prob

    def _log_prob(self, normal, raw):
        """
        Compute the log probability of a raw action after the action is squashed.
        Both inputs act on the raw underlying distribution.
        Because tanh_mean does not affect the density, we can ignore it.
        However, tanh_scale will affect the relative contribution of each component.'
        See Appendix C in the Soft Actor-Critic paper

        Args:
            normal (torch.distributions.normal.Normal): The "raw" normal distribution.
            raw (torch.Tensor): The "raw" action.

        Returns:
            torch.Tensor: The probability of the raw action, accounting for the affects of tanh.
        """
        log_prob = normal.log_prob(raw)
        log_prob -= torch.log(1 - torch.tanh(raw).pow(2) + 1e-06)
        log_prob -= torch.log(self._tanh_scale)
        return log_prob.sum(-1)

    def _squash(self, x):
        return torch.tanh(x) * self._tanh_scale + self._tanh_mean

    def to(self, device):
        self._tanh_mean = self._tanh_mean
        self._tanh_scale = self._tanh_scale
        return super()


class SoftmaxPolicyNetwork(RLNetwork):

    def __init__(self, model):
        super().__init__(model)

    def forward(self, state):
        outputs = super().forward(state)
        probs = functional.softmax(outputs, dim=-1)
        return torch.distributions.Categorical(probs)


class fc_policy(nn.Module):

    def __init__(self, env, hidden1=400, hidden2=300):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(env.state_space.shape[0] + 1, hidden1), nn.Tanh(), nn.Linear(hidden1, hidden2), nn.Tanh(), nn.Linear(hidden2, env.action_space.shape[0]))
        self.log_stds = nn.Parameter(torch.zeros(env.action_space.shape[0]))

    def forward(self, x):
        means = self.model(x)
        stds = self.log_stds.expand(*means.shape)
        return torch.cat((means, stds), 1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Aggregation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (CategoricalDueling,
     lambda: ([], {'value_model': _mock_layer(), 'advantage_model': _mock_layer()}),
     lambda: ([torch.rand([4, 1, 1])], {}),
     True),
    (Dueling,
     lambda: ([], {'value_model': _mock_layer(), 'advantage_model': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear0,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NoisyFactorizedLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NoisyLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Scale,
     lambda: ([], {'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_cpnota_autonomous_learning_library(_paritybench_base):
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

