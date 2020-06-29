import sys
_module = sys.modules[__name__]
del sys
all = _module
agents = _module
_agent = _module
a2c = _module
c51 = _module
ddpg = _module
ddqn = _module
dqn = _module
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
environments = _module
atari_wrappers = _module
gym = _module
state = _module
state_test = _module
experiments = _module
experiment = _module
parallel_env_experiment = _module
parallel_env_experiment_test = _module
plots = _module
run_experiment = _module
single_env_experiment = _module
single_env_experiment_test = _module
slurm = _module
watch = _module
writer = _module
logging = _module
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
softmax = _module
softmax_test = _module
presets = _module
dqn = _module
models = _module
test_ = _module
atari_test = _module
classic_control = _module
classic_control_test = _module
continuous = _module
continuous_test = _module
validate_agent = _module
atari40 = _module
pybullet = _module
conf = _module
examples = _module
slurm_experiment = _module
scripts = _module
classic = _module
plot = _module
release = _module
watch_atari = _module
watch_classic = _module
watch_continuous = _module
setup = _module

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


from torch.nn.functional import mse_loss


import torch


from torch.distributions.normal import Normal


from torch.nn import utils


from torch.nn import functional as F


from torch import nn


from torch.nn.functional import smooth_l1_loss


import numpy as np


from torch.nn import *


from torch.nn import functional


from torch.optim import Adam


from torch.optim.lr_scheduler import CosineAnnealingLR


DONE = torch.tensor([0], dtype=torch.uint8)


NOT_DONE = torch.tensor([1], dtype=torch.uint8)


class State:

    def __init__(self, raw, mask=None, info=None):
        self._raw = raw
        if mask is None:
            self._mask = torch.ones(len(raw), dtype=torch.uint8, device=raw
                .device)
        else:
            self._mask = mask
        self._info = info or [None] * len(raw)

    @classmethod
    def from_list(cls, states):
        raw = torch.cat([state.features for state in states])
        done = torch.cat([state.mask for state in states])
        info = sum([state.info for state in states], [])
        return cls(raw, done, info)

    @classmethod
    def from_gym(cls, numpy_arr, done, info, device='cpu', dtype=np.float32):
        raw = torch.from_numpy(np.array(numpy_arr, dtype=dtype)).unsqueeze(0
            ).to(device)
        mask = DONE.to(device) if done else NOT_DONE.to(device)
        return cls(raw, mask=mask, info=[info])

    @property
    def features(self):
        """
        Default features are the raw state.
        Override this method for other types of features.
        """
        return self._raw

    @property
    def mask(self):
        return self._mask

    @property
    def info(self):
        return self._info

    @property
    def raw(self):
        return self._raw

    @property
    def done(self):
        return not self._mask

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return State(self._raw[idx], self._mask[idx], self._info[idx])
        if isinstance(idx, torch.Tensor):
            return State(self._raw[idx], self._mask[idx])
        return State(self._raw[idx].unsqueeze(0), self._mask[idx].unsqueeze
            (0), [self._info[idx]])

    def __len__(self):
        return len(self._raw)


class FeatureModule(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, states):
        features = self.model(states.features.float())
        return State(features, mask=states.mask, info=states.info)


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
        values = self.model(states).view((len(states), self.n_actions, self
            .n_atoms))
        values = F.softmax(values, dim=2)
        values = (values - self.terminal) * states.mask.view((-1, 1, 1)).float(
            ) + self.terminal
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
        return self.model(state.features.float()) * state.mask.float(
            ).unsqueeze(-1)


class Aggregation(nn.Module):
    """len()
    Aggregation layer for the Dueling architecture.

    https://arxiv.org/abs/1511.06581
    This layer computes a Q function by combining
    an estimate of V with an estimate of the advantage.
    The advantage is normalized by substracting the average
    advantage so that we can propertly
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
        super(Dueling, self).__init__()
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
        super(CategoricalDueling, self).__init__()
        self.value_model = value_model
        self.advantage_model = advantage_model

    def forward(self, features):
        batch_size = len(features)
        value_dist = self.value_model(features)
        atoms = value_dist.shape[1]
        advantage_dist = self.advantage_model(features).view((batch_size, -
            1, atoms))
        advantage_mean = advantage_dist.mean(dim=1, keepdim=True)
        return (value_dist.view((batch_size, 1, atoms)) + advantage_dist -
            advantage_mean).view((batch_size, -1))


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
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features,
            in_features).fill_(sigma_init))
        self.register_buffer('epsilon_weight', torch.zeros(out_features,
            in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_
                (sigma_init))
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
        return F.linear(x, self.weight + self.sigma_weight * self.
            epsilon_weight, bias)


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise

    N.B. nn.Linear already initializes weight and bias to
    """

    def __init__(self, in_features, out_features, sigma_init=0.4,
        init_scale=3, bias=True):
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_init / np.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features,
            in_features).fill_(sigma_init))
        self.register_buffer('epsilon_input', torch.zeros(1, in_features))
        self.register_buffer('epsilon_output', torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_
                (sigma_init))

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
        self.register_buffer('weight', torch.tensor((action_space.high -
            action_space.low) / 2))
        self.register_buffer('bias', torch.tensor((action_space.high +
            action_space.low) / 2))

    def forward(self, x):
        return torch.tanh(x) * self.weight + self.bias


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_cpnota_autonomous_learning_library(_paritybench_base):
    pass
    def test_000(self):
        self._check(Aggregation(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(CategoricalDueling(*[], **{'value_model': ReLU(), 'advantage_model': ReLU()}), [torch.rand([4, 4])], {})

    def test_002(self):
        self._check(Dueling(*[], **{'value_model': ReLU(), 'advantage_model': ReLU()}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Linear0(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(NoisyFactorizedLinear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(NoisyLinear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(Scale(*[], **{'scale': 1.0}), [torch.rand([4, 4, 4, 4])], {})

