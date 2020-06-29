import sys
_module = sys.modules[__name__]
del sys
benchmark = _module
cherry = _module
_torch = _module
_utils = _module
_version = _module
algorithms = _module
a2c = _module
ddpg = _module
ppo = _module
sac = _module
trpo = _module
debug = _module
distributions = _module
envs = _module
action_lambda_wrapper = _module
action_space_scaler_wrapper = _module
base = _module
logger_wrapper = _module
monitor_wrapper = _module
normalizer_wrapper = _module
openai_atari_wrapper = _module
recorder_wrapper = _module
reward_clipper_wrapper = _module
reward_normalizer_wrapper = _module
runner_wrapper = _module
state_lambda_wrapper = _module
state_normalizer_wrapper = _module
timestep_wrapper = _module
torch_wrapper = _module
utils = _module
visdom_logger_wrapper = _module
experience_replay = _module
models = _module
atari = _module
robotics = _module
tabular = _module
utils = _module
nn = _module
epsilon_greedy = _module
init = _module
robotics_layers = _module
optim = _module
pg = _module
plot = _module
td = _module
actor_critic_cartpole = _module
actor_critic_gridworld = _module
a2c_atari = _module
debug_atari = _module
dist_a2c_atari = _module
dqn_atari = _module
ppo_atari = _module
bsuite = _module
trpo_v_random = _module
ppo_pendulum_gpu = _module
delayed_tsac_pybullet = _module
dist_ppo_pybullet = _module
ppo_pybullet = _module
sac_pybullet = _module
reinforce_cartpole = _module
simple_q_mdp = _module
cherry_ddpg = _module
cherry_dqn = _module
cherry_ppo = _module
cherry_sac = _module
cherry_vpg = _module
q_learning = _module
sarsa = _module
setup = _module
dummy_env = _module
integration = _module
actor_critic_tests = _module
spinup_ddpg_tests = _module
spinup_ppo_tests = _module
spinup_sac_tests = _module
spinup_vpg_tests = _module
unit = _module
_torch_tests = _module
algorithms_ppo_tests = _module
algorithms_trpo_tests = _module
base_wrapper_tests = _module
debug_logger_tests = _module
experience_replay_tests = _module
logger_wrapper_tests = _module
models_utils_tests = _module
plot_tests = _module
rl_tests = _module
runner_wrapper_tests = _module

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


from torch.nn import functional as F


import torch as th


from torch import autograd


from torch.nn.utils import parameters_to_vector


from torch.nn.utils import vector_to_parameters


import torch.nn as nn


from torch.distributions import Categorical


from torch.distributions import Normal


from torch.distributions import Distribution


from torch import nn


from torch.distributions import Bernoulli


import numpy as np


import random


from itertools import count


import torch.nn.functional as F


import torch.optim as optim


import torch.distributed as dist


import copy


from torch.distributions import kl_divergence


from copy import deepcopy


import torch


from torch import optim


import time


from collections import deque


class Reparameterization(object):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/distributions.py)

    **Description**

    Unifies interface for distributions that support `rsample` and those that do not.

    When calling `sample()`, this class checks whether `density` has a `rsample()` member,
    and defaults to call `sample()` if it does not.

    **References**

    1. Kingma and Welling. 2013. “Auto-Encoding Variational Bayes.” arXiv [stat.ML].

    **Arguments**

    * **density** (Distribution) - The distribution to wrap.

    **Example**

    ~~~python
    density = Normal(mean, std)
    reparam = Reparameterization(density)
    sample = reparam.sample()  # Uses Normal.rsample()
    ~~~

    """

    def __init__(self, density):
        self.density = density

    def sample(self, *args, **kwargs):
        if self.density.has_rsample:
            return self.density.rsample(*args, **kwargs)
        return self.density.sample(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.density, name)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'Reparameterization(' + str(self.density) + ')'


class ActionDistribution(nn.Module):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/distributions.py)

    **Description**

    A helper module to automatically choose the proper policy distribution,
    based on the Gym environment `action_space`.

    For `Discrete` action spaces, it uses a `Categorical` distribution, otherwise
    it uses a `Normal` which uses a diagonal covariance matrix.

    This class enables to write single version policy body that will be compatible
    with a variety of environments.

    **Arguments**

    * **env** (Environment) - Gym environment for which actions will be sampled.
    * **logstd** (float/tensor, *optional*, default=0) - The log standard
    deviation for the `Normal` distribution.
    * **use_probs** (bool, *optional*, default=False) - Whether to use probabilities or logits
    for the `Categorical` case.
    * **reparam** (bool, *optional*, default=False) - Whether to use reparameterization in the
    `Normal` case.

    **Example**

    ~~~python
    env = gym.make('CartPole-v1')
    action_dist = ActionDistribution(env)
    ~~~

    """

    def __init__(self, env, logstd=None, use_probs=False, reparam=False):
        super(ActionDistribution, self).__init__()
        self.use_probs = use_probs
        self.reparam = reparam
        self.is_discrete = ch.envs.is_discrete(env.action_space)
        if not self.is_discrete:
            if logstd is None:
                action_size = ch.envs.get_space_dimension(env.action_space,
                    vectorized_dims=False)
                logstd = nn.Parameter(th.zeros(action_size))
            if isinstance(logstd, (float, int)):
                logstd = nn.Parameter(th.Tensor([logstd]))
            self.logstd = logstd

    def forward(self, x):
        if self.is_discrete:
            if self.use_probs:
                return Categorical(probs=x)
            return Categorical(logits=x)
        else:
            density = Normal(loc=x, scale=self.logstd.exp())
            if self.reparam:
                density = Reparameterization(density)
            return density


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


def atari_init_(module, gain=None):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/nn/init.py)

    **Description**

    Default initialization for Atari environments.

    **Credit**

    Adapted from Ilya Kostrikov's implementation, itself inspired from OpenAI Baslines.

    **Arguments**

    * **module** (nn.Module) - Module to initialize.
    * **gain** (float, *optional*, default=None) - Gain of orthogonal initialization.
    Default is computed for ReLU activation with `torch.nn.init.calculate_gain('relu')`.

    **Returns**

    * Module, whose weight and bias have been modified in-place.

    **Example**

    ~~~python
    linear = nn.Linear(23, 5)
    atari_init_(linear)
    ~~~

    """
    if gain is None:
        gain = nn.init.calculate_gain('relu')
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0.0)
    return module


class NatureFeatures(nn.Sequential):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/models/atari.py)

    **Description**

    The convolutional body of the DQN architecture.

    **References**

    1. Mnih et al. 2015. “Human-Level Control through Deep Reinforcement Learning.”
    2. Mnih et al. 2016. “Asynchronous Methods for Deep Reinforcement Learning.”

    **Credit**

    Adapted from Ilya Kostrikov's implementation.

    **Arguments**

    * **input_size** (int) - Number of channels.
      (Stacked frames in original implementation.)
    * **output_size** (int, *optional*, default=512) - Size of the output
      representation.
    """

    def __init__(self, input_size=4, hidden_size=512):
        super(NatureFeatures, self).__init__(atari_init_(nn.Conv2d(
            input_size, 32, 8, stride=4, padding=0)), nn.ReLU(),
            atari_init_(nn.Conv2d(32, 64, 4, stride=2, padding=0)), nn.ReLU
            (), atari_init_(nn.Conv2d(64, 32, 3, stride=1, padding=0)), nn.
            ReLU(), Flatten(), atari_init_(nn.Linear(32 * 7 * 7,
            hidden_size)), nn.ReLU())


class NatureActor(nn.Linear):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/models/atari.py)

    **Description**

    The actor head of the A3C architecture.

    **References**

    1. Mnih et al. 2015. “Human-Level Control through Deep Reinforcement Learning.”
    2. Mnih et al. 2016. “Asynchronous Methods for Deep Reinforcement Learning.”

    **Credit**

    Adapted from Ilya Kostrikov's implementation.

    **Arguments**

    * **input_size** (int) - Size of input of the fully connected layers
    * **output_size** (int) - Size of the action space.
    """

    def __init__(self, input_size, output_size):
        super(NatureActor, self).__init__(input_size, output_size)
        atari_init_(self, gain=0.01)


class NatureCritic(nn.Linear):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/models/atari.py)

    **Description**

    The critic head of the A3C architecture.

    **References**

    1. Mnih et al. 2015. “Human-Level Control through Deep Reinforcement Learning.”
    2. Mnih et al. 2016. “Asynchronous Methods for Deep Reinforcement Learning.”

    **Credit**

    Adapted from Ilya Kostrikov's implementation.

    **Arguments**

    * **input_size** (int) - Size of input of the fully connected layers
    * **output_size** (int, *optional*, default=1) - Size of the value.
    """

    def __init__(self, input_size, output_size=1):
        super(NatureCritic, self).__init__(input_size, output_size)
        atari_init_(self, gain=1.0)


class RoboticsMLP(nn.Module):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/models/robotics.py)

    **Description**

    A multi-layer perceptron with proper initialization for robotic control.

    **Credit**

    Adapted from Ilya Kostrikov's implementation.

    **Arguments**

    * **inputs_size** (int) - Size of input.
    * **output_size** (int) - Size of output.
    * **layer_sizes** (list, *optional*, default=None) - A list of ints,
      each indicating the size of a hidden layer.
      (Defaults to two hidden layers of 64 units.)

    **Example**
    ~~~python
    target_qf = ch.models.robotics.RoboticsMLP(23,
                                             34,
                                             layer_sizes=[32, 32])
    ~~~
    """

    def __init__(self, input_size, output_size, layer_sizes=None):
        super(RoboticsMLP, self).__init__()
        if layer_sizes is None:
            layer_sizes = [64, 64]
        if len(layer_sizes) > 0:
            layers = [RoboticsLinear(input_size, layer_sizes[0]), nn.Tanh()]
            for in_, out_ in zip(layer_sizes[:-1], layer_sizes[1:]):
                layers.append(RoboticsLinear(in_, out_))
                layers.append(nn.Tanh())
            layers.append(RoboticsLinear(layer_sizes[-1], output_size))
        else:
            layers = [RoboticsLinear(input_size, output_size)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LinearValue(nn.Module):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/models/robotics.py)

    **Description**

    A linear state-value function, whose parameters are found by minimizing
    least-squares.

    **Credit**

    Adapted from Tristan Deleu's implementation.

    **References**

    1. Duan et al. 2016. “Benchmarking Deep Reinforcement Learning for Continuous Control.”
    2. [https://github.com/tristandeleu/pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl)

    **Arguments**

    * **inputs_size** (int) - Size of input.
    * **reg** (float, *optional*, default=1e-5) - Regularization coefficient.

    **Example**
    ~~~python
    states = replay.state()
    rewards = replay.reward()
    dones = replay.done()
    returns = ch.td.discount(gamma, rewards, dones)
    baseline = LinearValue(input_size)
    baseline.fit(states, returns)
    next_values = baseline(replay.next_states())
    ~~~
    """

    def __init__(self, input_size, reg=1e-05):
        super(LinearValue, self).__init__()
        self.linear = nn.Linear(2 * input_size + 4, 1, bias=False)
        self.reg = reg

    def _features(self, states):
        length = states.size(0)
        ones = th.ones(length, 1)
        al = th.arange(length, dtype=th.float32, device=states.device).view(
            -1, 1) / 100.0
        return th.cat([states, states ** 2, al, al ** 2, al ** 3, ones], dim=1)

    def fit(self, states, returns):
        features = self._features(states)
        reg = self.reg * th.eye(features.size(1))
        reg = reg
        A = features.t() @ features + reg
        b = features.t() @ returns
        if hasattr(th, 'lstsq'):
            coeffs, _ = th.lstsq(b, A)
        else:
            coeffs, _ = th.gels(b, A)
        self.linear.weight.data = coeffs.data.t()

    def forward(self, states):
        features = self._features(states)
        return self.linear(features)


class StateValueFunction(nn.Module):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/models/tabular.py)

    **Description**

    Stores a table of state values, V(s), one for each state.

    Assumes that the states are one-hot encoded.
    Also, the returned values are differentiable and can be used in
    conjunction with PyTorch's optimizers.

    **Arguments**

    * **state_size** (int) - The number of states in the environment.
    * **init** (function, *optional*, default=None) - The initialization
      scheme for the values in the table. (Default is 0.)

    **References**

    1. Sutton, Richard, and Andrew Barto. 2018. Reinforcement Learning, Second Edition. The MIT Press.

    **Example**
    ~~~python
    vf = StateValueFunction(env.state_size)
    state = env.reset()
    state = ch.onehot(state, env.state_size)
    state_value = vf(state)
    ~~~

    """

    def __init__(self, state_size, init=None):
        super(StateValueFunction, self).__init__()
        self.values = nn.Parameter(th.zeros((state_size, 1)))
        self.state_size = state_size
        if init is not None:
            if isinstance(init, (float, int, th.Tensor)):
                self.values.data.add_(init)
            else:
                init(self.values)

    def forward(self, state):
        return state.view(-1, self.state_size) @ self.values


class ActionValueFunction(nn.Module):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/models/tabular.py)

    **Description**

    Stores a table of action values, Q(s, a), one for each
    (state, action) pair.

    Assumes that the states and actions are one-hot encoded.
    Also, the returned values are differentiable and can be used in
    conjunction with PyTorch's optimizers.

    **Arguments**

    * **state_size** (int) - The number of states in the environment.
    * **action_size** (int) - The number of actions per state.
    * **init** (function, *optional*, default=None) - The initialization
      scheme for the values in the table. (Default is 0.)

    **References**

    1. Sutton, Richard, and Andrew Barto. 2018. Reinforcement Learning, Second Edition. The MIT Press.

    **Example**
    ~~~python
    qf = ActionValueFunction(env.state_size, env.action_size)
    state = env.reset()
    state = ch.onehot(state, env.state_size)
    all_action_values = qf(state)
    action = ch.onehot(0, env.action_size)
    action_value = qf(state, action)
    ~~~

    """

    def __init__(self, state_size, action_size, init=None):
        super(ActionValueFunction, self).__init__()
        self.values = nn.Parameter(th.zeros((state_size, action_size),
            requires_grad=True))
        self.state_size = state_size
        self.action_size = action_size
        if init is not None:
            if isinstance(init, (float, int, th.Tensor)):
                self.values.data.add_(init)
            else:
                init(self.values)

    def forward(self, state, action=None):
        action_values = (state @ self.values).view(-1, self.action_size)
        if action is None:
            return action_values
        return th.sum(action * action_values, dim=1, keepdim=True)


class RandomPolicy(nn.Module):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/models/utils.py)

    **Description**

    Policy that randomly samples actions from the environment action space.

    **Arguments**

    * **env** (Environment) - Environment from which to sample actions.

    **Example**
    ~~~python
    policy = ch.models.RandomPolicy(env)
    env = envs.Runner(env)
    replay = env.run(policy, steps=2048)
    ~~~
    """

    def __init__(self, env, *args, **kwargs):
        super(RandomPolicy, self).__init__()
        self.env = env

    def forward(self, *args, **kwargs):
        return self.env.action_space.sample()


class EpsilonGreedy(nn.Module):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/nn/epsilon_greedy.py)

    **Description**

    Samples actions from a uniform distribution with probability `epsilon` or
    the one maximizing the input with probability `1 - epsilon`.

    **References**

    1. Sutton, Richard, and Andrew Barto. 2018. Reinforcement Learning, Second Edition. The MIT Press.

    **Arguments**

    * **epsilon** (float, *optional*, default=0.05) - The epsilon factor.
    * **learnable** (bool, *optional*, default=False) - Whether the epsilon
    factor is a learnable parameter or not.

    **Example**

    ~~~python
    egreedy = EpsilonGreedy()
    q_values = q_value(state)  # NxM tensor
    actions = egreedy(q_values)  # Nx1 tensor of longs
    ~~~

    """

    def __init__(self, epsilon=0.05, learnable=False):
        super(EpsilonGreedy, self).__init__()
        msg = 'EpsilonGreedy: epsilon is not in a valid range.'
        assert epsilon >= 0.0 and epsilon <= 1.0, msg
        if learnable:
            epsilon = nn.Parameter(th.Tensor([epsilon]))
        self.epsilon = epsilon

    def forward(self, x):
        bests = x.max(dim=1, keepdim=True)[1]
        sampled = Categorical(probs=th.ones_like(x)).sample()
        probs = th.ones(x.size(0), 1) - self.epsilon
        b = Bernoulli(probs=probs).sample().long()
        ret = bests * b + (1 - b) * sampled
        return ret


class RoboticsLinear(nn.Linear):
    """
    [[Source]](https://github.com/seba-1511/cherry/blob/master/cherry/nn/robotics_layers.py)

    **Description**

    Akin to `nn.Linear`, but with proper initialization for robotic control.

    **Credit**

    Adapted from Ilya Kostrikov's implementation.

    **Arguments**


    * **gain** (float, *optional*) - Gain factor passed to `robotics_init_` initialization.
    * This class extends `nn.Linear` and supports all of its arguments.

    **Example**

    ~~~python
    linear = ch.nn.Linear(23, 5, bias=True)
    action_mean = linear(state)
    ~~~

    """

    def __init__(self, *args, **kwargs):
        gain = kwargs.pop('gain', None)
        super(RoboticsLinear, self).__init__(*args, **kwargs)
        ch.nn.init.robotics_init_(self, gain=gain)


class ActorCriticNet(nn.Module):

    def __init__(self, env):
        super(ActorCriticNet, self).__init__()
        self.affine1 = nn.Linear(env.state_size, 128)
        self.action_head = nn.Linear(128, env.action_size)
        self.value_head = nn.Linear(128, 1)
        self.distribution = distributions.ActionDistribution(env, use_probs
            =True)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        action_mass = self.distribution(F.softmax(action_scores, dim=1))
        value = self.value_head(x)
        return action_mass, value


class ActorCriticNet(nn.Module):

    def __init__(self, env):
        super(ActorCriticNet, self).__init__()
        self.affine1 = nn.Linear(env.state_size['image'], 128)
        self.action_head = nn.Linear(128, env.action_size)
        self.value_head = nn.Linear(128, 1)
        self.distribution = distributions.ActionDistribution(env)

    def forward(self, x):
        x = x.view(-1)
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        action_mass = self.distribution(F.log_softmax(action_scores, dim=0))
        value = self.value_head(x)
        return action_mass, value


class NatureCNN(nn.Module):

    def __init__(self, env, hidden_size=512):
        super(NatureCNN, self).__init__()
        self.input_size = 4
        self.features = atari.NatureFeatures(self.input_size, hidden_size)
        self.critic = atari.NatureCritic(hidden_size)
        self.actor = atari.NatureActor(hidden_size, env.action_size)
        self.action_dist = distributions.ActionDistribution(env, use_probs=
            False)

    def forward(self, x):
        x = x.view(-1, self.input_size, 84, 84).mul(1 / 255.0)
        features = self.features(x)
        value = self.critic(features)
        density = self.actor(features)
        mass = self.action_dist(density)
        return mass, value


class NatureCNN(nn.Module):

    def __init__(self, env, hidden_size=512):
        super(NatureCNN, self).__init__()
        self.input_size = 4
        self.features = atari.NatureFeatures(self.input_size, hidden_size)
        self.critic = atari.NatureCritic(hidden_size)
        self.actor = atari.NatureActor(hidden_size, env.action_size)
        self.action_dist = distributions.ActionDistribution(env, use_probs=
            False)

    def forward(self, x):
        x = x.view(-1, self.input_size, 84, 84).mul(1 / 255.0)
        features = self.features(x)
        value = self.critic(features)
        density = self.actor(features)
        mass = self.action_dist(density)
        return mass, value


class NatureCNN(nn.Module):

    def __init__(self, env, hidden_size=512):
        super(NatureCNN, self).__init__()
        self.input_size = 4
        self.features = atari.NatureFeatures(self.input_size, hidden_size)
        self.critic = atari.NatureCritic(hidden_size)
        self.actor = atari.NatureActor(hidden_size, env.action_size)
        self.action_dist = distributions.ActionDistribution(env, use_probs=
            False)

    def forward(self, x):
        x = x.view(-1, self.input_size, 84, 84).mul(1 / 255.0)
        features = self.features(x)
        value = self.critic(features)
        density = self.actor(features)
        mass = self.action_dist(density)
        return mass, value


class DQN(nn.Module):

    def __init__(self, env, hidden_size=512):
        super(DQN, self).__init__()
        self.input_size = 4
        self.features = atari.NatureFeatures(self.input_size, hidden_size)
        self.q = atari.NatureActor(hidden_size, env.action_size)

    def forward(self, x):
        x = x.view(-1, self.input_size, 84, 84).mul(1 / 255.0)
        features = self.features(x)
        q_values = self.q(features)
        return q_values


class NatureCNN(nn.Module):

    def __init__(self, env, hidden_size=512):
        super(NatureCNN, self).__init__()
        self.input_size = 4
        self.features = atari.NatureFeatures(self.input_size, hidden_size)
        self.critic = atari.NatureCritic(hidden_size)
        self.actor = atari.NatureActor(hidden_size, env.action_size)
        self.action_dist = distributions.ActionDistribution(env, use_probs=
            False)

    def forward(self, x):
        x = x.view(-1, self.input_size, 84, 84).mul(1 / 255.0)
        features = self.features(x)
        value = self.critic(features)
        density = self.actor(features)
        mass = self.action_dist(density)
        return mass, value


class Policy(nn.Module):

    def __init__(self, env):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(env.state_size, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, env.action_size)
        self.dist = ch.distributions.ActionDistribution(env)

    def density(self, state):
        x = self.layer1(state)
        x = self.relu(x)
        x = self.layer2(x)
        return self.dist(x)

    def log_prob(self, state, action):
        density = self.density(state)
        return density.log_prob(action)

    def forward(self, state):
        density = self.density(state)
        return density.sample().detach()


class Actor(nn.Module):

    def __init__(self, hidden_size, stochastic=True, layer_norm=False):
        super().__init__()
        layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(
            hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [
                nn.LayerNorm(hidden_size)] + layers[3:]
        self.policy = nn.Sequential(*layers)
        if stochastic:
            self.policy_log_std = nn.Parameter(torch.tensor([[0.0]]))

    def forward(self, state):
        policy = self.policy(state)
        return policy


class Critic(nn.Module):

    def __init__(self, hidden_size, state_action=False, layer_norm=False):
        super().__init__()
        self.state_action = state_action
        layers = [nn.Linear(3 + (1 if state_action else 0), hidden_size),
            nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.
            Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [
                nn.LayerNorm(hidden_size)] + layers[3:]
        self.value = nn.Sequential(*layers)

    def forward(self, state, action=None):
        if self.state_action:
            value = self.value(torch.cat([state, action], dim=1))
        else:
            value = self.value(state)
        return value.squeeze(dim=1)


class ActorCritic(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.actor = Actor(hidden_size, stochastic=True)
        self.critic = Critic(hidden_size)

    def forward(self, state):
        policy = Normal(self.actor(state), self.actor.policy_log_std.exp())
        value = self.critic(state)
        action = policy.sample()
        log_prob = policy.log_prob(action)
        return action, {'mass': policy, 'log_prob': log_prob, 'value': value}


class MLP(nn.Module):

    def __init__(self, input_size, output_size, layer_sizes=None, init_w=0.003
        ):
        super(MLP, self).__init__()
        if layer_sizes is None:
            layer_sizes = [300, 300]
        self.layers = nn.ModuleList()
        in_size = input_size
        for next_size in layer_sizes:
            fc = nn.Linear(in_size, next_size)
            self.layers.append(fc)
            in_size = next_size
        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, *args, **kwargs):
        h = th.cat(args, dim=1)
        for fc in self.layers:
            h = F.relu(fc(h))
        output = self.last_fc(h)
        return output


class ActorCriticNet(nn.Module):

    def __init__(self, env):
        super(ActorCriticNet, self).__init__()
        self.actor = models.robotics.Actor(env.state_size, env.action_size,
            layer_sizes=[64, 64])
        self.critic = models.robotics.RoboticsMLP(env.state_size, 1)
        self.action_dist = distributions.ActionDistribution(env, use_probs=
            False, reparam=False)

    def forward(self, x):
        action_scores = self.actor(x)
        action_density = self.action_dist(action_scores)
        value = self.critic(x)
        return action_density, value


class ActorCriticNet(nn.Module):

    def __init__(self, env):
        super(ActorCriticNet, self).__init__()
        self.actor = models.robotics.RoboticsActor(env.state_size, env.
            action_size, layer_sizes=[64, 64])
        self.critic = models.robotics.RoboticsMLP(env.state_size, 1)
        self.action_dist = dist.ActionDistribution(env, use_probs=False,
            reparam=False)

    def forward(self, x):
        action_scores = self.actor(x)
        action_density = self.action_dist(action_scores)
        value = self.critic(x)
        return action_density, value


class MLP(nn.Module):

    def __init__(self, input_size, output_size, layer_sizes=None, init_w=0.003
        ):
        super(MLP, self).__init__()
        if layer_sizes is None:
            layer_sizes = [300, 300]
        self.layers = nn.ModuleList()
        in_size = input_size
        for next_size in layer_sizes:
            fc = nn.Linear(in_size, next_size)
            self.layers.append(fc)
            in_size = next_size
        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, *args, **kwargs):
        h = th.cat(args, dim=1)
        for fc in self.layers:
            h = F.relu(fc(h))
        output = self.last_fc(h)
        return output


class PolicyNet(nn.Module):

    def __init__(self):
        super(PolicyNet, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden=128):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden, bias=False)
        self.linear2 = nn.Linear(hidden, output_size, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = th.tanh(x)
        x = self.linear2(x)
        x = th.tanh(x)
        return x


class Actor(nn.Module):

    def __init__(self, hidden_size, stochastic=True, layer_norm=False):
        super().__init__()
        layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(
            hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [
                nn.LayerNorm(hidden_size)] + layers[3:]
        self.policy = nn.Sequential(*layers)
        if stochastic:
            self.policy_log_std = nn.Parameter(torch.tensor([[0.0]]))

    def forward(self, state):
        policy = self.policy(state)
        return policy


class Critic(nn.Module):

    def __init__(self, hidden_size, state_action=False, layer_norm=False):
        super().__init__()
        self.state_action = state_action
        layers = [nn.Linear(3 + (1 if state_action else 0), hidden_size),
            nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.
            Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [
                nn.LayerNorm(hidden_size)] + layers[3:]
        self.value = nn.Sequential(*layers)

    def forward(self, state, action=None):
        if self.state_action:
            value = self.value(torch.cat([state, action], dim=1))
        else:
            value = self.value(state)
        return value.squeeze(dim=1)


EPSILON = 0.05


class DQN(nn.Module):

    def __init__(self, hidden_size, num_actions=5):
        super().__init__()
        layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(
            hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size,
            num_actions)]
        self.dqn = nn.Sequential(*layers)
        self.egreedy = ch.nn.EpsilonGreedy(EPSILON)

    def forward(self, state):
        values = self.dqn(state)
        action = self.egreedy(values)
        return action, values


class Actor(nn.Module):

    def __init__(self, hidden_size, stochastic=True, layer_norm=False):
        super().__init__()
        layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(
            hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [
                nn.LayerNorm(hidden_size)] + layers[3:]
        self.policy = nn.Sequential(*layers)
        if stochastic:
            self.policy_log_std = nn.Parameter(torch.tensor([[0.0]]))

    def forward(self, state):
        policy = self.policy(state)
        return policy


class Critic(nn.Module):

    def __init__(self, hidden_size, state_action=False, layer_norm=False):
        super().__init__()
        self.state_action = state_action
        layers = [nn.Linear(3 + (1 if state_action else 0), hidden_size),
            nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.
            Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [
                nn.LayerNorm(hidden_size)] + layers[3:]
        self.value = nn.Sequential(*layers)

    def forward(self, state, action=None):
        if self.state_action:
            value = self.value(torch.cat([state, action], dim=1))
        else:
            value = self.value(state)
        return value.squeeze(dim=1)


class ActorCritic(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.actor = Actor(hidden_size, stochastic=True)
        self.critic = Critic(hidden_size)

    def forward(self, state):
        policy = Normal(self.actor(state), self.actor.policy_log_std.exp())
        value = self.critic(state)
        action = policy.sample()
        log_prob = policy.log_prob(action)
        return action, {'mass': policy, 'log_prob': log_prob, 'value': value}


class TanhNormal(Distribution):

    def __init__(self, loc, scale):
        super().__init__()
        self.normal = Normal(loc, scale)

    def sample(self):
        return torch.tanh(self.normal.sample())

    def rsample(self):
        return torch.tanh(self.normal.rsample())

    def log_prob(self, value):
        inv_value = (torch.log1p(value) - torch.log1p(-value)) / 2
        return self.normal.log_prob(inv_value) - torch.log1p(-value.pow(2) +
            1e-06)

    @property
    def mean(self):
        return torch.tanh(self.normal.mean)


class SoftActor(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.log_std_min, self.log_std_max = -20, 2
        layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(
            hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 2)]
        self.policy = nn.Sequential(*layers)

    def forward(self, state):
        policy_mean, policy_log_std = self.policy(state).chunk(2, dim=1)
        policy_log_std = torch.clamp(policy_log_std, min=self.log_std_min,
            max=self.log_std_max)
        policy = TanhNormal(policy_mean, policy_log_std.exp())
        return policy


class Critic(nn.Module):

    def __init__(self, hidden_size, state_action=False, layer_norm=False):
        super().__init__()
        self.state_action = state_action
        layers = [nn.Linear(3 + (1 if state_action else 0), hidden_size),
            nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.
            Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [
                nn.LayerNorm(hidden_size)] + layers[3:]
        self.value = nn.Sequential(*layers)

    def forward(self, state, action=None):
        if self.state_action:
            value = self.value(torch.cat([state, action], dim=1))
        else:
            value = self.value(state)
        return value.squeeze(dim=1)


class Actor(nn.Module):

    def __init__(self, hidden_size, stochastic=True, layer_norm=False):
        super().__init__()
        layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(
            hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [
                nn.LayerNorm(hidden_size)] + layers[3:]
        self.policy = nn.Sequential(*layers)
        if stochastic:
            self.policy_log_std = nn.Parameter(torch.tensor([[0.0]]))

    def forward(self, state):
        policy = self.policy(state)
        return policy


class Critic(nn.Module):

    def __init__(self, hidden_size, state_action=False, layer_norm=False):
        super().__init__()
        self.state_action = state_action
        layers = [nn.Linear(3 + (1 if state_action else 0), hidden_size),
            nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.
            Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [
                nn.LayerNorm(hidden_size)] + layers[3:]
        self.value = nn.Sequential(*layers)

    def forward(self, state, action=None):
        if self.state_action:
            value = self.value(torch.cat([state, action], dim=1))
        else:
            value = self.value(state)
        return value.squeeze(dim=1)


class ActorCritic(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.actor = Actor(hidden_size, stochastic=True)
        self.critic = Critic(hidden_size)

    def forward(self, state):
        policy = Normal(self.actor(state), self.actor.policy_log_std.exp())
        value = self.critic(state)
        action = policy.sample()
        log_prob = policy.log_prob(action)
        return action, {'log_prob': log_prob, 'value': value}


class Agent(nn.Module):

    def __init__(self, env):
        super(Agent, self).__init__()
        self.env = env
        self.qf = ActionValueFunction(env.state_size, env.action_size)
        self.e_greedy = ch.nn.EpsilonGreedy(0.1)

    def forward(self, x):
        x = ch.onehot(x, self.env.state_size)
        q_values = self.qf(x)
        action = self.e_greedy(q_values)
        info = {'q_action': q_values[:, (action)]}
        return action, info


class Agent(nn.Module):

    def __init__(self, env):
        super(Agent, self).__init__()
        self.env = env
        self.qf = ActionValueFunction(env.state_size, env.action_size)
        self.e_greedy = ch.nn.EpsilonGreedy(0.1)

    def forward(self, x):
        x = ch.onehot(x, self.env.state_size)
        q_values = self.qf(x)
        action = self.e_greedy(q_values)
        info = {'q_action': q_values[:, (action)]}
        return action, info


class ActorCriticNet(nn.Module):

    def __init__(self, env):
        super(ActorCriticNet, self).__init__()
        self.affine = nn.Linear(env.state_size, 128)
        self.action_head = nn.Linear(128, env.action_size)
        self.value_head = nn.Linear(128, 1)
        self.distribution = distributions.ActionDistribution(env, use_probs
            =True)

    def forward(self, x):
        x = F.relu(self.affine(x))
        action_scores = self.action_head(x)
        action_mass = self.distribution(F.softmax(action_scores, dim=1))
        value = self.value_head(x)
        return action_mass, value


class Actor(nn.Module):

    def __init__(self, hidden_size, stochastic=True, layer_norm=False):
        super().__init__()
        layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(
            hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [
                nn.LayerNorm(hidden_size)] + layers[3:]
        self.policy = nn.Sequential(*layers)
        if stochastic:
            self.policy_log_std = nn.Parameter(torch.tensor([[0.0]]))

    def forward(self, state):
        policy = self.policy(state)
        return policy


class Critic(nn.Module):

    def __init__(self, hidden_size, state_action=False, layer_norm=False):
        super().__init__()
        self.state_action = state_action
        layers = [nn.Linear(3 + (1 if state_action else 0), hidden_size),
            nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.
            Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [
                nn.LayerNorm(hidden_size)] + layers[3:]
        self.value = nn.Sequential(*layers)

    def forward(self, state, action=None):
        if self.state_action:
            value = self.value(torch.cat([state, action], dim=1))
        else:
            value = self.value(state)
        return value.squeeze(dim=1)


class Actor(nn.Module):

    def __init__(self, hidden_size, stochastic=True, layer_norm=False):
        super().__init__()
        layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(
            hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [
                nn.LayerNorm(hidden_size)] + layers[3:]
        self.policy = nn.Sequential(*layers)
        if stochastic:
            self.policy_log_std = nn.Parameter(torch.tensor([[0.0]]))

    def forward(self, state):
        policy = self.policy(state)
        return policy


class Critic(nn.Module):

    def __init__(self, hidden_size, state_action=False, layer_norm=False):
        super().__init__()
        self.state_action = state_action
        layers = [nn.Linear(3 + (1 if state_action else 0), hidden_size),
            nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.
            Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [
                nn.LayerNorm(hidden_size)] + layers[3:]
        self.value = nn.Sequential(*layers)

    def forward(self, state, action=None):
        if self.state_action:
            value = self.value(torch.cat([state, action], dim=1))
        else:
            value = self.value(state)
        return value.squeeze(dim=1)


class ActorCritic(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.actor = Actor(hidden_size, stochastic=True)
        self.critic = Critic(hidden_size)

    def forward(self, state):
        policy = Normal(self.actor(state), self.actor.policy_log_std.exp())
        value = self.critic(state)
        return policy, value


class SoftActor(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.log_std_min, self.log_std_max = -20, 2
        layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(
            hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 2)]
        self.policy = nn.Sequential(*layers)

    def forward(self, state):
        policy_mean, policy_log_std = self.policy(state).chunk(2, dim=1)
        policy_log_std = torch.clamp(policy_log_std, min=self.log_std_min,
            max=self.log_std_max)
        policy = TanhNormal(policy_mean, policy_log_std.exp())
        return policy


class Critic(nn.Module):

    def __init__(self, hidden_size, state_action=False, layer_norm=False):
        super().__init__()
        self.state_action = state_action
        layers = [nn.Linear(3 + (1 if state_action else 0), hidden_size),
            nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.
            Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [
                nn.LayerNorm(hidden_size)] + layers[3:]
        self.value = nn.Sequential(*layers)

    def forward(self, state, action=None):
        if self.state_action:
            value = self.value(torch.cat([state, action], dim=1))
        else:
            value = self.value(state)
        return value.squeeze(dim=1)


class Actor(nn.Module):

    def __init__(self, hidden_size, stochastic=True, layer_norm=False):
        super().__init__()
        layers = [nn.Linear(3, hidden_size), nn.Tanh(), nn.Linear(
            hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [
                nn.LayerNorm(hidden_size)] + layers[3:]
        self.policy = nn.Sequential(*layers)
        if stochastic:
            self.policy_log_std = nn.Parameter(torch.tensor([[0.0]]))

    def forward(self, state):
        policy = self.policy(state)
        return policy


class Critic(nn.Module):

    def __init__(self, hidden_size, state_action=False, layer_norm=False):
        super().__init__()
        self.state_action = state_action
        layers = [nn.Linear(3 + (1 if state_action else 0), hidden_size),
            nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.
            Linear(hidden_size, 1)]
        if layer_norm:
            layers = layers[:1] + [nn.LayerNorm(hidden_size)] + layers[1:3] + [
                nn.LayerNorm(hidden_size)] + layers[3:]
        self.value = nn.Sequential(*layers)

    def forward(self, state, action=None):
        if self.state_action:
            value = self.value(torch.cat([state, action], dim=1))
        else:
            value = self.value(state)
        return value.squeeze(dim=1)


class ActorCritic(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.actor = Actor(hidden_size, stochastic=True)
        self.critic = Critic(hidden_size)

    def forward(self, state):
        policy = Normal(self.actor(state), self.actor.policy_log_std.exp())
        value = self.critic(state)
        return policy, value


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_learnables_cherry(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ActionValueFunction(*[], **{'state_size': 4, 'action_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Actor(*[], **{'hidden_size': 4}), [torch.rand([3, 3])], {})

    @_fails_compile()
    def test_002(self):
        self._check(ActorCritic(*[], **{'hidden_size': 4}), [torch.rand([3, 3])], {})

    @_fails_compile()
    def test_003(self):
        self._check(Critic(*[], **{'hidden_size': 4}), [torch.rand([3, 3])], {})

    @_fails_compile()
    def test_004(self):
        self._check(EpsilonGreedy(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(LinearValue(*[], **{'input_size': 4}), [torch.rand([4, 4])], {})

    def test_007(self):
        self._check(MLP(*[], **{'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(NatureActor(*[], **{'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(NatureCritic(*[], **{'input_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(PolicyNet(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_011(self):
        self._check(SoftActor(*[], **{'hidden_size': 4}), [torch.rand([3, 3])], {})

    def test_012(self):
        self._check(StateValueFunction(*[], **{'state_size': 4}), [torch.rand([4, 4, 4, 4])], {})

