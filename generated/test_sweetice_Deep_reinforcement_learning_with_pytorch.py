import sys
_module = sys.modules[__name__]
del sys
Sarsa = _module
gridworld = _module
DQN = _module
DQN_mountain_car_v1 = _module
naiveDQN = _module
PolicyGradient = _module
REINFORCE = _module
REINFORCE_with_Baseline = _module
Run_Model = _module
A2C = _module
multiprocessing_env = _module
DDPG = _module
PPO2 = _module
PPO_CartPole_v0 = _module
PPO_pendulum = _module
SAC = _module
SAC_dual_Q_net = _module
test_agent = _module
TD3 = _module
plot = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import copy


from collections import namedtuple


from itertools import count


import torch.optim as optim


from torch.distributions import Normal


from torch.distributions import Categorical


from torch.utils.data.sampler import BatchSampler


from torch.utils.data.sampler import SubsetRandomSampler


from torch import optim


from copy import deepcopy


from torch.nn.utils import clip_grad_norm_


from torch.nn import functional as F


from torch.optim import adam


from torch.autograd import Variable


from torch.optim import Adam


from torch.nn.functional import smooth_l1_loss


import math


import random


from torch.autograd import grad


_global_config['env_name'] = 4


class Net(nn.Module):
    """docstring for Net"""

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, 30)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, NUM_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


class Policy(nn.Module):

    def __init__(self, n_states, n_hidden, n_output):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(n_states, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_output)
        self.reward = []
        self.log_act_probs = []
        self.Gt = []
        self.sigma = []

    def forward(self, x):
        x = F.relu(self.linear1(x))
        output = F.softmax(self.linear2(x), dim=1)
        return output


gamma = 0.99


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_space, 20)
        self.fc3 = nn.Linear(20, action_space)
        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_space, 32)
        self.action_head = nn.Linear(32, action_space)
        self.value_head = nn.Linear(32, 1)
        self.save_actions = []
        self.rewards = []
        os.makedirs('./AC_CartPole-v0', exist_ok=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_score = self.action_head(x)
        state_value = self.value_head(x)
        return F.softmax(action_score, dim=-1), state_value


GAMMA = 0.995


class Module(nn.Module):

    def __init__(self):
        super(Module, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.action_head = nn.Linear(128, num_action)
        self.value_head = nn.Linear(128, 1)
        self.policy_action_value = []
        self.rewards = []
        self.gamma = GAMMA
        os.makedirs('/AC_MountainCar-v0_Model/', exist_ok=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        probs = F.softmax(self.action_head(x))
        value = self.value_head(x)
        return probs, value


class ActorCritic(nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        self.critic = nn.Sequential(nn.Linear(num_inputs, hidden_size), nn.
            ReLU(), nn.Linear(hidden_size, 1))
        self.actor = nn.Sequential(nn.Linear(num_inputs, hidden_size), nn.
            ReLU(), nn.Linear(hidden_size, num_outputs), nn.Softmax(dim=1))

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64, 8)
        self.mu_head = nn.Linear(8, 1)
        self.sigma_head = nn.Linear(8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        mu = self.mu_head(x)
        sigma = self.sigma_head(x)
        return mu, sigma


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64, 8)
        self.state_value = nn.Linear(8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        value = self.state_value(x)
        return value


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.action_head = nn.Linear(128, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.state_value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class ActorNet(nn.Module):

    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc = nn.Linear(3, 100)
        self.mu_head = nn.Linear(100, 1)
        self.sigma_head = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        mu = 2.0 * F.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return mu, sigma


class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(3, 100)
        self.v_head = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        state_value = self.v_head(x)
        return state_value


class Critic(nn.Module):

    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Critic(nn.Module):

    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):

    def __init__(self, state_dim, min_log_std=-10, max_log_std=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, 1)
        self.log_std_head = nn.Linear(256, 1)
        self.max_action = max_action
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.
            max_log_std)
        return mu, log_std_head


class Critic(nn.Module):

    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Q(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, s, a):
        s = s.reshape(-1, state_dim)
        a = a.reshape(-1, action_dim)
        x = torch.cat((s, a), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Critic(nn.Module):

    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Q(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, s, a):
        s = s.reshape(-1, state_dim)
        a = a.reshape(-1, action_dim)
        x = torch.cat((s, a), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.max_action
        return a


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.max_action
        return a


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_sweetice_Deep_reinforcement_learning_with_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ActorCritic(*[], **{'num_inputs': 4, 'num_outputs': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Actor(*[], **{'state_dim': 4, 'action_dim': 4, 'max_action': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Critic(*[], **{'state_dim': 4, 'action_dim': 4}), [torch.rand([4, 4, 4, 8]), torch.rand([4, 4, 4, 8])], {})

    def test_003(self):
        self._check(ActorNet(*[], **{}), [torch.rand([3, 3])], {})

    def test_004(self):
        self._check(CriticNet(*[], **{}), [torch.rand([3, 3])], {})

