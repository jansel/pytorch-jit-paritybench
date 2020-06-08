import sys
_module = sys.modules[__name__]
del sys
fireup = _module
algos = _module
core = _module
ddpg = _module
core = _module
dqn = _module
core = _module
ppo = _module
core = _module
sac = _module
core = _module
td3 = _module
core = _module
trpo = _module
vpg = _module
core = _module
vpg = _module
bench_ppo_cartpole = _module
bench_vpg_cartpole = _module
run = _module
user_config = _module
utils = _module
logx = _module
mpi_tools = _module
mpi_torch = _module
plot = _module
run_entrypoint = _module
run_utils = _module
serialization_utils = _module
test_policy = _module
version = _module
setup = _module
test_ppo = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.distributions.categorical import Categorical


from torch.distributions.normal import Normal


import scipy.signal


from torch.nn.utils import parameters_to_vector


from torch.distributions.kl import kl_divergence


from torch.nn.utils import vector_to_parameters


class MLP(nn.Module):

    def __init__(self, layers, activation=torch.tanh, output_activation=
        None, output_scale=1, output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_scale = output_scale
        self.output_squeeze = output_squeeze
        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x) * self.output_scale
        else:
            x = self.output_activation(self.layers[-1](x)) * self.output_scale
        return x.squeeze() if self.output_squeeze else x


class ActorCritic(nn.Module):

    def __init__(self, in_features, action_space, hidden_sizes=(400, 300),
        activation=torch.relu, output_activation=torch.tanh):
        super(ActorCritic, self).__init__()
        action_dim = action_space.shape[0]
        action_scale = action_space.high[0]
        self.policy = MLP(layers=[in_features] + list(hidden_sizes) + [
            action_dim], activation=activation, output_activation=
            output_activation, output_scale=action_scale)
        self.q = MLP(layers=[in_features + action_dim] + list(hidden_sizes) +
            [1], activation=activation, output_squeeze=True)

    def forward(self, x, a):
        pi = self.policy(x)
        q = self.q(torch.cat((x, a), dim=1))
        q_pi = self.q(torch.cat((x, pi), dim=1))
        return pi, q, q_pi


class MLP(nn.Module):

    def __init__(self, layers, activation=torch.tanh, output_activation=
        None, output_scale=1, output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_scale = output_scale
        self.output_squeeze = output_squeeze
        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x) * self.output_scale
        else:
            x = self.output_activation(self.layers[-1](x)) * self.output_scale
        return x.squeeze() if self.output_squeeze else x


class DQNetwork(nn.Module):

    def __init__(self, in_features, action_space, hidden_sizes=(400, 300),
        activation=torch.relu, output_activation=None):
        super(DQNetwork, self).__init__()
        action_dim = action_space.n
        self.q = MLP(layers=[in_features] + list(hidden_sizes) + [
            action_dim], activation=activation, output_activation=
            output_activation)

    def forward(self, x):
        return self.q(x)

    def policy(self, x):
        return torch.argmax(self.q(x), dim=1, keepdim=True)


class MLP(nn.Module):

    def __init__(self, layers, activation=torch.tanh, output_activation=
        None, output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze
        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))
        return x.squeeze() if self.output_squeeze else x


class CategoricalPolicy(nn.Module):

    def __init__(self, in_features, hidden_sizes, activation,
        output_activation, action_dim):
        super(CategoricalPolicy, self).__init__()
        self.logits = MLP(layers=[in_features] + list(hidden_sizes) + [
            action_dim], activation=activation)

    def forward(self, x, a=None):
        logits = self.logits(x)
        policy = Categorical(logits=logits)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()
        if a is not None:
            logp = policy.log_prob(a).squeeze()
        else:
            logp = None
        return pi, logp, logp_pi


class GaussianPolicy(nn.Module):

    def __init__(self, in_features, hidden_sizes, activation,
        output_activation, action_dim):
        super(GaussianPolicy, self).__init__()
        self.mu = MLP(layers=[in_features] + list(hidden_sizes) + [
            action_dim], activation=activation, output_activation=
            output_activation)
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim, dtype=
            torch.float32))

    def forward(self, x, a=None):
        mu = self.mu(x)
        policy = Normal(mu, self.log_std.exp())
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).sum(dim=1)
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else:
            logp = None
        return pi, logp, logp_pi


class ActorCritic(nn.Module):

    def __init__(self, in_features, action_space, hidden_sizes=(64, 64),
        activation=torch.tanh, output_activation=None, policy=None):
        super(ActorCritic, self).__init__()
        if policy is None and isinstance(action_space, Box):
            self.policy = GaussianPolicy(in_features, hidden_sizes,
                activation, output_activation, action_dim=action_space.shape[0]
                )
        elif policy is None and isinstance(action_space, Discrete):
            self.policy = CategoricalPolicy(in_features, hidden_sizes,
                activation, output_activation, action_dim=action_space.n)
        else:
            self.policy = policy(in_features, hidden_sizes, activation,
                output_activation, action_space)
        self.value_function = MLP(layers=[in_features] + list(hidden_sizes) +
            [1], activation=activation, output_squeeze=True)

    def forward(self, x, a=None):
        pi, logp, logp_pi = self.policy(x, a)
        v = self.value_function(x)
        return pi, logp, logp_pi, v


class MLP(nn.Module):

    def __init__(self, layers, activation=torch.tanh, output_activation=
        None, output_scale=1, output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_scale = output_scale
        self.output_squeeze = output_squeeze
        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x) * self.output_scale
        else:
            x = self.output_activation(self.layers[-1](x)) * self.output_scale
        return x.squeeze() if self.output_squeeze else x


LOG_STD_MAX = 2


EPS = 1e-08


LOG_STD_MIN = -20


class GaussianPolicy(nn.Module):

    def __init__(self, in_features, hidden_sizes, activation,
        output_activation, action_space):
        super(GaussianPolicy, self).__init__()
        action_dim = action_space.shape[0]
        self.action_scale = action_space.high[0]
        self.output_activation = output_activation
        self.net = MLP(layers=[in_features] + list(hidden_sizes),
            activation=activation, output_activation=activation)
        self.mu = nn.Linear(in_features=list(hidden_sizes)[-1],
            out_features=action_dim)
        """
        Because this algorithm maximizes trade-off of reward and entropy,
        entropy must be unique to state---and therefore log_stds need
        to be a neural network output instead of a shared-across-states
        learnable parameter vector. But for deep Relu and other nets,
        simply sticking an activationless dense layer at the end would
        be quite bad---at the beginning of training, a randomly initialized
        net could produce extremely large values for the log_stds, which
        would result in some actions being either entirely deterministic
        or too random to come back to earth. Either of these introduces
        numerical instability which could break the algorithm. To
        protect against that, we'll constrain the output range of the
        log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is
        slightly different from the trick used by the original authors of
        SAC---they used torch.clamp instead of squashing and rescaling.
        I prefer this approach because it allows gradient propagation
        through log_std where clipping wouldn't, but I don't know if
        it makes much of a difference.
        """
        self.log_std = nn.Sequential(nn.Linear(in_features=list(
            hidden_sizes)[-1], out_features=action_dim), nn.Tanh())

    def forward(self, x):
        output = self.net(x)
        mu = self.mu(output)
        if self.output_activation:
            mu = self.output_activation(mu)
        log_std = self.log_std(output)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std +
            1)
        policy = Normal(mu, torch.exp(log_std))
        pi = policy.rsample()
        logp_pi = torch.sum(policy.log_prob(pi), dim=1)
        mu, pi, logp_pi = self._apply_squashing_func(mu, pi, logp_pi)
        mu_scaled = mu * self.action_scale
        pi_scaled = pi * self.action_scale
        return pi_scaled, mu_scaled, logp_pi

    def _clip_but_pass_gradient(self, x, l=-1.0, u=1.0):
        clip_up = (x > u).float()
        clip_low = (x < l).float()
        return x + ((u - x) * clip_up + (l - x) * clip_low).detach()

    def _apply_squashing_func(self, mu, pi, logp_pi):
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        logp_pi -= torch.sum(torch.log(self._clip_but_pass_gradient(1 - pi **
            2, l=0, u=1) + EPS), dim=1)
        return mu, pi, logp_pi


class ActorCritic(nn.Module):

    def __init__(self, in_features, action_space, hidden_sizes=(400, 300),
        activation=torch.relu, output_activation=None, policy=GaussianPolicy):
        super(ActorCritic, self).__init__()
        action_dim = action_space.shape[0]
        self.policy = policy(in_features, hidden_sizes, activation,
            output_activation, action_space)
        self.vf_mlp = MLP([in_features] + list(hidden_sizes) + [1],
            activation, output_squeeze=True)
        self.q1 = MLP([in_features + action_dim] + list(hidden_sizes) + [1],
            activation, output_squeeze=True)
        self.q2 = MLP([in_features + action_dim] + list(hidden_sizes) + [1],
            activation, output_squeeze=True)

    def forward(self, x, a):
        pi, mu, logp_pi = self.policy(x)
        q1 = self.q1(torch.cat((x, a), dim=1))
        q1_pi = self.q1(torch.cat((x, pi), dim=1))
        q2 = self.q2(torch.cat((x, a), dim=1))
        q2_pi = self.q2(torch.cat((x, pi), dim=1))
        v = self.vf_mlp(x)
        return pi, mu, logp_pi, q1, q2, q1_pi, q2_pi, v


class MLP(nn.Module):

    def __init__(self, layers, activation=torch.tanh, output_activation=
        None, output_scale=1, output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_scale = output_scale
        self.output_squeeze = output_squeeze
        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x) * self.output_scale
        else:
            x = self.output_activation(self.layers[-1](x)) * self.output_scale
        return torch.squeeze(x) if self.output_squeeze else x


class ActorCritic(nn.Module):

    def __init__(self, in_features, action_space, hidden_sizes=(400, 300),
        activation=torch.relu, output_activation=torch.tanh):
        super(ActorCritic, self).__init__()
        action_dim = action_space.shape[0]
        action_scale = action_space.high[0]
        self.policy = MLP(layers=[in_features] + list(hidden_sizes) + [
            action_dim], activation=activation, output_activation=
            output_activation, output_scale=action_scale)
        self.q1 = MLP(layers=[in_features + action_dim] + list(hidden_sizes
            ) + [1], activation=activation, output_squeeze=True)
        self.q2 = MLP(layers=[in_features + action_dim] + list(hidden_sizes
            ) + [1], activation=activation, output_squeeze=True)

    def forward(self, x, a):
        pi = self.policy(x)
        q1 = self.q1(torch.cat((x, a), dim=1))
        q2 = self.q2(torch.cat((x, a), dim=1))
        q1_pi = self.q1(torch.cat((x, pi), dim=1))
        return pi, q1, q2, q1_pi


class MLP(nn.Module):

    def __init__(self, layers, activation=torch.tanh, output_activation=
        None, output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze
        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))
        return x.squeeze() if self.output_squeeze else x


class CategoricalPolicy(nn.Module):

    def __init__(self, in_features, hidden_sizes, activation,
        output_activation, action_dim):
        super(CategoricalPolicy, self).__init__()
        self.logits = MLP(layers=[in_features] + list(hidden_sizes) + [
            action_dim], activation=activation)

    def forward(self, x, a=None, old_logits=None):
        logits = self.logits(x)
        policy = Categorical(logits=logits)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()
        if a is not None:
            logp = policy.log_prob(a).squeeze()
        else:
            logp = None
        if old_logits is not None:
            old_policy = Categorical(logits=old_logits)
            d_kl = kl_divergence(old_policy, policy).mean()
        else:
            d_kl = None
        info = {'old_logits': logits.detach().numpy()}
        return pi, logp, logp_pi, info, d_kl


class GaussianPolicy(nn.Module):

    def __init__(self, in_features, hidden_sizes, activation,
        output_activation, action_dim):
        super(GaussianPolicy, self).__init__()
        self.mu = MLP(layers=[in_features] + list(hidden_sizes) + [
            action_dim], activation=activation, output_activation=
            output_activation)
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim, dtype=
            torch.float32))

    def forward(self, x, a=None, old_log_std=None, old_mu=None):
        mu = self.mu(x)
        policy = Normal(mu, self.log_std.exp())
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).sum(dim=1)
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else:
            logp = None
        if old_mu is not None or old_log_std is not None:
            old_policy = Normal(old_mu, old_log_std.exp())
            d_kl = kl_divergence(old_policy, policy).mean()
        else:
            d_kl = None
        info = {'old_mu': np.squeeze(mu.detach().numpy()), 'old_log_std':
            self.log_std.detach().numpy()}
        return pi, logp, logp_pi, info, d_kl


class ActorCritic(nn.Module):

    def __init__(self, in_features, action_space, hidden_sizes=(64, 64),
        activation=torch.tanh, output_activation=None, policy=None):
        super(ActorCritic, self).__init__()
        if policy is None and isinstance(action_space, Box):
            self.policy = GaussianPolicy(in_features, hidden_sizes,
                activation, output_activation, action_dim=action_space.shape[0]
                )
        elif policy is None and isinstance(action_space, Discrete):
            self.policy = CategoricalPolicy(in_features, hidden_sizes,
                activation, output_activation, action_dim=action_space.n)
        else:
            self.policy = policy(in_features, hidden_sizes, activation,
                output_activation, action_space)
        self.value_function = MLP(layers=[in_features] + list(hidden_sizes) +
            [1], activation=activation, output_squeeze=True)

    def forward(self, x, a=None, **kwargs):
        pi, logp, logp_pi, info, d_kl = self.policy(x, a, **kwargs)
        v = self.value_function(x)
        return pi, logp, logp_pi, info, d_kl, v


class MLP(nn.Module):

    def __init__(self, layers, activation=torch.tanh, output_activation=
        None, output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze
        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))
        return x.squeeze() if self.output_squeeze else x


class CategoricalPolicy(nn.Module):

    def __init__(self, in_features, hidden_sizes, activation,
        output_activation, action_dim):
        super(CategoricalPolicy, self).__init__()
        self.logits = MLP(layers=[in_features] + list(hidden_sizes) + [
            action_dim], activation=activation)

    def forward(self, x, a=None):
        logits = self.logits(x)
        policy = Categorical(logits=logits)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()
        if a is not None:
            logp = policy.log_prob(a).squeeze()
        else:
            logp = None
        return pi, logp, logp_pi


class GaussianPolicy(nn.Module):

    def __init__(self, in_features, hidden_sizes, activation,
        output_activation, action_dim):
        super(GaussianPolicy, self).__init__()
        self.mu = MLP(layers=[in_features] + list(hidden_sizes) + [
            action_dim], activation=activation, output_activation=
            output_activation)
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim))

    def forward(self, x, a=None):
        policy = Normal(self.mu(x), self.log_std.exp())
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).sum(dim=1)
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else:
            logp = None
        return pi, logp, logp_pi


class ActorCritic(nn.Module):

    def __init__(self, in_features, action_space, hidden_sizes=(64, 64),
        activation=torch.tanh, output_activation=None, policy=None):
        super(ActorCritic, self).__init__()
        if policy is None and isinstance(action_space, Box):
            self.policy = GaussianPolicy(in_features, hidden_sizes,
                activation, output_activation, action_dim=action_space.shape[0]
                )
        elif policy is None and isinstance(action_space, Discrete):
            self.policy = CategoricalPolicy(in_features, hidden_sizes,
                activation, output_activation, action_dim=action_space.n)
        else:
            self.policy = policy(in_features, hidden_sizes, activation,
                output_activation, action_space)
        self.value_function = MLP(layers=[in_features] + list(hidden_sizes) +
            [1], activation=activation, output_squeeze=True)

    def forward(self, x, a=None):
        pi, logp, logp_pi = self.policy(x, a)
        v = self.value_function(x)
        return pi, logp, logp_pi, v


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_kashif_firedup(_paritybench_base):
    pass
    @_fails_compile()

    def test_000(self):
        self._check(MLP(*[], **{'layers': [4, 4]}), [torch.rand([4, 4, 4, 4])], {})
