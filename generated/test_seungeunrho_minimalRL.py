import sys
_module = sys.modules[__name__]
del sys
REINFORCE = _module
a2c = _module
a3c = _module
acer = _module
actor_critic = _module
ddpg = _module
dqn = _module
ppo = _module
sac = _module
vtrace = _module

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


import torch.nn.functional as F


import torch.optim as optim


from torch.distributions import Categorical


import torch.multiprocessing as mp


import numpy as np


import time


import random


import collections


from torch.distributions import Normal


gamma = 0.98


learning_rate = 0.0005


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []


class ActorCritic(nn.Module):

    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


class MuNet(nn.Module):

    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * 2
        return mu


tau = 0.01


class QNet(nn.Module):

    def __init__(self, learning_rate):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc_cat = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a), target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


class Qnet(nn.Module):

    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


K_epoch = 3


eps_clip = 0.1


lmbda = 0.95


class PPO(nn.Module):

    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


init_alpha = 0.01


lr_alpha = 0.001


target_entropy = -1.0


class PolicyNet(nn.Module):

    def __init__(self, learning_rate):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc_mu = nn.Linear(128, 1)
        self.fc_std = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-07)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s, a), q2(s, a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        loss = -min_q - entropy
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


clip_c_threshold = 1.0


clip_rho_threshold = 1.0


class Vtrace(nn.Module):

    def __init__(self):
        super(Vtrace, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.clip_rho_threshold = torch.tensor(clip_rho_threshold, dtype=torch.float)
        self.clip_c_threshold = torch.tensor(clip_c_threshold, dtype=torch.float)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, mu_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, mu_a, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            mu_a_lst.append([mu_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        s, a, r, s_prime, done_mask, mu_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), torch.tensor(done_lst, dtype=torch.float), torch.tensor(mu_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, mu_a

    def vtrace(self, s, a, r, s_prime, done_mask, mu_a):
        with torch.no_grad():
            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            v, v_prime = self.v(s), self.v(s_prime)
            ratio = torch.exp(torch.log(pi_a) - torch.log(mu_a))
            rhos = torch.min(self.clip_rho_threshold, ratio)
            cs = torch.min(self.clip_c_threshold, ratio).numpy()
            td_target = r + gamma * v_prime * done_mask
            delta = rhos * (td_target - v).numpy()
            vs_minus_v_xs_lst = []
            vs_minus_v_xs = 0.0
            vs_minus_v_xs_lst.append([vs_minus_v_xs])
            for i in range(len(delta) - 1, -1, -1):
                vs_minus_v_xs = gamma * cs[i][0] * vs_minus_v_xs + delta[i][0]
                vs_minus_v_xs_lst.append([vs_minus_v_xs])
            vs_minus_v_xs_lst.reverse()
            vs_minus_v_xs = torch.tensor(vs_minus_v_xs_lst, dtype=torch.float)
            vs = vs_minus_v_xs[:-1] + v.numpy()
            vs_prime = vs_minus_v_xs[1:] + v_prime.numpy()
            advantage = r + gamma * vs_prime - v.numpy()
        return vs, advantage, rhos

    def train_net(self):
        s, a, r, s_prime, done_mask, mu_a = self.make_batch()
        vs, advantage, rhos = self.vtrace(s, a, r, s_prime, done_mask, mu_a)
        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        val_loss = F.smooth_l1_loss(self.v(s), vs)
        pi_loss = -rhos * torch.log(pi_a) * advantage
        loss = pi_loss + val_loss
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Policy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Qnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_seungeunrho_minimalRL(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

