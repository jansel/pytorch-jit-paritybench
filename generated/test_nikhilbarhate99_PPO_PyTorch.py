import sys
_module = sys.modules[__name__]
del sys
PPO = _module
PPO_continuous = _module
test = _module
test_continuous = _module

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


from torch.distributions import Categorical


from torch.distributions import MultivariateNormal


import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        self.action_layer = nn.Sequential(nn.Linear(state_dim, n_latent_var
            ), nn.Tanh(), nn.Linear(n_latent_var, n_latent_var), nn.Tanh(),
            nn.Linear(n_latent_var, action_dim), nn.Softmax(dim=-1))
        self.value_layer = nn.Sequential(nn.Linear(state_dim, n_latent_var),
            nn.Tanh(), nn.Linear(n_latent_var, n_latent_var), nn.Tanh(), nn
            .Linear(n_latent_var, 1))

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float()
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.value_layer(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class ActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.
            Linear(64, 32), nn.Tanh(), nn.Linear(32, action_dim), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn
            .Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
        self.action_var = torch.full((action_dim,), action_std * action_std)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_nikhilbarhate99_PPO_PyTorch(_paritybench_base):
    pass
