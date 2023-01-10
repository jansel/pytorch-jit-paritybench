import sys
_module = sys.modules[__name__]
del sys
a2c_ppo_acktr = _module
algo = _module
a2c_acktr = _module
gail = _module
kfac = _module
ppo = _module
arguments = _module
distributions = _module
envs = _module
model = _module
storage = _module
utils = _module
enjoy = _module
evaluation = _module
convert_to_pytorch = _module
generate_tmux_yaml = _module
main = _module
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


import torch


import torch.nn as nn


import torch.optim as optim


import numpy as np


import torch.nn.functional as F


import torch.utils.data


from torch import autograd


import math


from torch.utils.data.sampler import BatchSampler


from torch.utils.data.sampler import SubsetRandomSampler


import copy


import time


from collections import deque


class Discriminator(nn.Module):

    def __init__(self, input_dim, hidden_dim, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.trunk.train()
        self.optimizer = torch.optim.Adam(self.trunk.parameters())
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self, expert_state, expert_action, policy_state, policy_action, lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)
        alpha = alpha.expand_as(expert_data)
        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True
        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size())
        grad = autograd.grad(outputs=disc, inputs=mixup_data, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()
        policy_data_generator = rollouts.feed_forward_generator(None, mini_batch_size=expert_loader.batch_size)
        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader, policy_data_generator):
            policy_state, policy_action = policy_batch[0], policy_batch[2]
            policy_d = self.trunk(torch.cat([policy_state, policy_action], dim=1))
            expert_state, expert_action = expert_batch
            expert_state = obsfilt(expert_state.numpy(), update=False)
            expert_state = torch.FloatTensor(expert_state)
            expert_action = expert_action
            expert_d = self.trunk(torch.cat([expert_state, expert_action], dim=1))
            expert_loss = F.binary_cross_entropy_with_logits(expert_d, torch.ones(expert_d.size()))
            policy_loss = F.binary_cross_entropy_with_logits(policy_d, torch.zeros(policy_d.size()))
            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action, policy_state, policy_action)
            loss += (gail_loss + grad_pen).item()
            n += 1
            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return loss / n

    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()
            if self.returns is None:
                self.returns = reward.clone()
            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())
            return reward / np.sqrt(self.ret_rms.var[0] + 1e-08)


class AddBias(nn.Module):

    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias


class SplitBias(nn.Module):

    def __init__(self, module):
        super(SplitBias, self).__init__()
        self.module = module
        self.add_bias = AddBias(module.bias.data)
        self.module.bias = None

    def forward(self, input):
        x = self.module(input)
        x = self.add_bias(x)
        return x


class FixedCategorical(torch.distributions.Categorical):

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Categorical(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class FixedNormal(torch.distributions.Normal):

    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class DiagGaussian(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros
        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class FixedBernoulli(torch.distributions.Bernoulli):

    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Bernoulli(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()
        self._hidden_size = hidden_size
        self._recurrent = recurrent
        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            N = hxs.size(0)
            T = int(x.size(0) / N)
            x = x.view(T, N, x.size(1))
            masks = masks.view(T, N)
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()
            has_zeros = [0] + has_zeros + [T]
            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                rnn_scores, hxs = self.gru(x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1))
                outputs.append(rnn_scores)
            x = torch.cat(outputs, dim=0)
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)
        return x, hxs


class CNNBase(NNBase):

    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        self.main = nn.Sequential(init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(), init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(), init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(), init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):

    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)
        if recurrent:
            num_inputs = hidden_size
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.actor = nn.Sequential(init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(), init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        self.critic = nn.Sequential(init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(), init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)
        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class Policy(nn.Module):

    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError
        self.base = base(obs_shape[0], **base_kwargs)
        if action_space.__class__.__name__ == 'Discrete':
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == 'Box':
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == 'MultiBinary':
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return value, action_log_probs, dist_entropy, rnn_hxs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Bernoulli,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Categorical,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DiagGaussian,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLPBase,
     lambda: ([], {'num_inputs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_ikostrikov_pytorch_a2c_ppo_acktr_gail(_paritybench_base):
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

