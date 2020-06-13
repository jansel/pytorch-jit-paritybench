import sys
_module = sys.modules[__name__]
del sys
a2c_agent = _module
arguments = _module
demo = _module
models = _module
train = _module
utils = _module
ddpg_agent = _module
models = _module
dqn_agent = _module
models = _module
models = _module
ppo_agent = _module
models = _module
sac_agent = _module
models = _module
trpo_agent = _module
rl_utils = _module
env_wrapper = _module
atari_wrapper = _module
create_env = _module
frame_stack = _module
multi_envs_wrapper = _module
experience_replay = _module
logger = _module
bench = _module
plot = _module
mpi_utils = _module
normalizer = _module
running_filter = _module
seeds = _module
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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch import nn


from torch.nn import functional as F


from torch import optim


import copy


class deepmind(nn.Module):

    def __init__(self):
        super(deepmind, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 512)
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.
            calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.
            calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.
            calculate_gain('relu'))
        nn.init.orthogonal_(self.fc1.weight.data, gain=nn.init.
            calculate_gain('relu'))
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        nn.init.constant_(self.fc1.bias.data, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        return x


class net(nn.Module):

    def __init__(self, num_actions):
        super(net, self).__init__()
        self.cnn_layer = deepmind()
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, num_actions)
        nn.init.orthogonal_(self.critic.weight.data)
        nn.init.constant_(self.critic.bias.data, 0)
        nn.init.orthogonal_(self.actor.weight.data, gain=0.01)
        nn.init.constant_(self.actor.bias.data, 0)

    def forward(self, inputs):
        x = self.cnn_layer(inputs / 255.0)
        value = self.critic(x)
        pi = F.softmax(self.actor(x), dim=1)
        return value, pi


class actor(nn.Module):

    def __init__(self, obs_dims, action_dims):
        super(actor, self).__init__()
        self.fc1 = nn.Linear(obs_dims, 400)
        self.fc2 = nn.Linear(400, 300)
        self.action_out = nn.Linear(300, action_dims)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = torch.tanh(self.action_out(x))
        return actions


class critic(nn.Module):

    def __init__(self, obs_dims, action_dims):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(obs_dims, 400)
        self.fc2 = nn.Linear(400 + action_dims, 300)
        self.q_out = nn.Linear(300, 1)

    def forward(self, x, actions):
        x = F.relu(self.fc1(x))
        x = torch.cat([x, actions], dim=1)
        x = F.relu(self.fc2(x))
        q_value = self.q_out(x)
        return q_value


class deepmind(nn.Module):

    def __init__(self):
        super(deepmind, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.
            calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.
            calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.
            calculate_gain('relu'))
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 7 * 7)
        return x


class net(nn.Module):

    def __init__(self, num_actions, use_dueling=False):
        super(net, self).__init__()
        self.use_dueling = use_dueling
        self.cnn_layer = deepmind()
        if not self.use_dueling:
            self.fc1 = nn.Linear(32 * 7 * 7, 256)
            self.action_value = nn.Linear(256, num_actions)
        else:
            self.action_fc = nn.Linear(32 * 7 * 7, 256)
            self.state_value_fc = nn.Linear(32 * 7 * 7, 256)
            self.action_value = nn.Linear(256, num_actions)
            self.state_value = nn.Linear(256, 1)

    def forward(self, inputs):
        x = self.cnn_layer(inputs / 255.0)
        if not self.use_dueling:
            x = F.relu(self.fc1(x))
            action_value_out = self.action_value(x)
        else:
            action_fc = F.relu(self.action_fc(x))
            action_value = self.action_value(action_fc)
            state_value_fc = F.relu(self.state_value_fc(x))
            state_value = self.state_value(state_value_fc)
            action_value_mean = torch.mean(action_value, dim=1, keepdim=True)
            action_value_center = action_value - action_value_mean
            action_value_out = state_value + action_value_center
        return action_value_out


class mlp_net(nn.Module):

    def __init__(self, state_size, num_actions, dist_type):
        super(mlp_net, self).__init__()
        self.dist_type = dist_type
        self.fc1_v = nn.Linear(state_size, 64)
        self.fc2_v = nn.Linear(64, 64)
        self.fc1_a = nn.Linear(state_size, 64)
        self.fc2_a = nn.Linear(64, 64)
        if self.dist_type == 'gauss':
            self.sigma_log = nn.Parameter(torch.zeros(1, num_actions))
            self.action_mean = nn.Linear(64, num_actions)
            self.action_mean.weight.data.mul_(0.1)
            self.action_mean.bias.data.zero_()
        elif self.dist_type == 'beta':
            self.action_alpha = nn.Linear(64, num_actions)
            self.action_beta = nn.Linear(64, num_actions)
            self.action_alpha.weight.data.mul_(0.1)
            self.action_alpha.bias.data.zero_()
            self.action_beta.weight.data.mul_(0.1)
            self.action_beta.bias.data.zero_()
        self.value = nn.Linear(64, 1)
        self.value.weight.data.mul_(0.1)
        self.value.bias.data.zero_()

    def forward(self, x):
        x_v = torch.tanh(self.fc1_v(x))
        x_v = torch.tanh(self.fc2_v(x_v))
        state_value = self.value(x_v)
        x_a = torch.tanh(self.fc1_a(x))
        x_a = torch.tanh(self.fc2_a(x_a))
        if self.dist_type == 'gauss':
            mean = self.action_mean(x_a)
            sigma_log = self.sigma_log.expand_as(mean)
            sigma = torch.exp(sigma_log)
            pi = mean, sigma
        elif self.dist_type == 'beta':
            alpha = F.softplus(self.action_alpha(x_a)) + 1
            beta = F.softplus(self.action_beta(x_a)) + 1
            pi = alpha, beta
        return state_value, pi


class deepmind(nn.Module):

    def __init__(self):
        super(deepmind, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 512)
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.
            calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.
            calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.
            calculate_gain('relu'))
        nn.init.orthogonal_(self.fc1.weight.data, gain=nn.init.
            calculate_gain('relu'))
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        nn.init.constant_(self.fc1.bias.data, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        return x


class cnn_net(nn.Module):

    def __init__(self, num_actions):
        super(cnn_net, self).__init__()
        self.cnn_layer = deepmind()
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, num_actions)
        nn.init.orthogonal_(self.critic.weight.data)
        nn.init.constant_(self.critic.bias.data, 0)
        nn.init.orthogonal_(self.actor.weight.data, gain=0.01)
        nn.init.constant_(self.actor.bias.data, 0)

    def forward(self, inputs):
        x = self.cnn_layer(inputs / 255.0)
        value = self.critic(x)
        pi = F.softmax(self.actor(x), dim=1)
        return value, pi


class flatten_mlp(nn.Module):

    def __init__(self, input_dims, hidden_size, action_dims=None):
        super(flatten_mlp, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_size
            ) if action_dims is None else nn.Linear(input_dims +
            action_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_value = nn.Linear(hidden_size, 1)

    def forward(self, obs, action=None):
        inputs = torch.cat([obs, action], dim=1) if action is not None else obs
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        output = self.q_value(x)
        return output


class tanh_gaussian_actor(nn.Module):

    def __init__(self, input_dims, action_dims, hidden_size, log_std_min,
        log_std_max):
        super(tanh_gaussian_actor, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dims)
        self.log_std = nn.Linear(hidden_size, action_dims)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.
            log_std_max)
        return mean, torch.exp(log_std)


class network(nn.Module):

    def __init__(self, num_states, num_actions):
        super(network, self).__init__()
        self.critic = critic(num_states)
        self.actor = actor(num_states, num_actions)

    def forward(self, x):
        state_value = self.critic(x)
        pi = self.actor(x)
        return state_value, pi


class critic(nn.Module):

    def __init__(self, num_states):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(num_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        value = self.value(x)
        return value


class actor(nn.Module):

    def __init__(self, num_states, num_actions):
        super(actor, self).__init__()
        self.fc1 = nn.Linear(num_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.action_mean = nn.Linear(64, num_actions)
        self.sigma_log = nn.Parameter(torch.zeros(1, num_actions))

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        mean = self.action_mean(x)
        sigma_log = self.sigma_log.expand_as(mean)
        sigma = torch.exp(sigma_log)
        pi = mean, sigma
        return pi


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_TianhongDai_reinforcement_learning_algorithms(_paritybench_base):
    pass
    def test_000(self):
        self._check(actor(*[], **{'num_states': 4, 'num_actions': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(critic(*[], **{'num_states': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(flatten_mlp(*[], **{'input_dims': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(network(*[], **{'num_states': 4, 'num_actions': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(tanh_gaussian_actor(*[], **{'input_dims': 4, 'action_dims': 4, 'hidden_size': 4, 'log_std_min': 4, 'log_std_max': 4}), [torch.rand([4, 4, 4, 4])], {})

