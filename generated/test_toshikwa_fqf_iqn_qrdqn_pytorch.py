import sys
_module = sys.modules[__name__]
del sys
fqf_iqn_qrdqn = _module
agent = _module
base_agent = _module
fqf_agent = _module
iqn_agent = _module
qrdqn_agent = _module
env = _module
memory = _module
base = _module
per = _module
segment_tree = _module
model = _module
base_model = _module
fqf = _module
iqn = _module
qrdqn = _module
network = _module
utils = _module
train_fqf = _module
train_iqn = _module
train_qrdqn = _module

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


import numpy as np


import torch


from torch.utils.tensorboard import SummaryWriter


from torch.optim import Adam


from torch.optim import RMSprop


from collections import deque


from torch import nn


from copy import copy


import torch.nn.functional as F


class NoisyLinear(nn.Module):

    def __init__(self, in_features, out_features, sigma=0.5):
        super(NoisyLinear, self).__init__()
        self.mu_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.sigma_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('eps_p', torch.FloatTensor(in_features))
        self.register_buffer('eps_q', torch.FloatTensor(out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma
        self.reset()
        self.sample()

    def reset(self):
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.out_features))

    def f(self, x):
        return x.normal_().sign().mul(x.abs().sqrt())

    def sample(self):
        self.eps_p.copy_(self.f(self.eps_p))
        self.eps_q.copy_(self.f(self.eps_q))

    def forward(self, x):
        if self.training:
            weight = self.mu_W + self.sigma_W * self.eps_q.ger(self.eps_p)
            bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()
        else:
            weight = self.mu_W
            bias = self.mu_bias
        return F.linear(x, weight, bias)


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def sample_noise(self):
        if self.noisy_net:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.sample()


class CosineEmbeddingNetwork(nn.Module):

    def __init__(self, num_cosines=64, embedding_dim=7 * 7 * 64, noisy_net=False):
        super(CosineEmbeddingNetwork, self).__init__()
        linear = NoisyLinear if noisy_net else nn.Linear
        self.net = nn.Sequential(linear(num_cosines, embedding_dim), nn.ReLU())
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus):
        batch_size = taus.shape[0]
        N = taus.shape[1]
        i_pi = np.pi * torch.arange(start=1, end=self.num_cosines + 1, dtype=taus.dtype, device=taus.device).view(1, 1, self.num_cosines)
        cosines = torch.cos(taus.view(batch_size, N, 1) * i_pi).view(batch_size * N, self.num_cosines)
        tau_embeddings = self.net(cosines).view(batch_size, N, self.embedding_dim)
        return tau_embeddings


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class DQNBase(nn.Module):

    def __init__(self, num_channels, embedding_dim=7 * 7 * 64):
        super(DQNBase, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(), Flatten()).apply(initialize_weights_he)
        self.embedding_dim = embedding_dim

    def forward(self, states):
        batch_size = states.shape[0]
        state_embedding = self.net(states)
        assert state_embedding.shape == (batch_size, self.embedding_dim)
        return state_embedding


class QuantileNetwork(nn.Module):

    def __init__(self, num_actions, embedding_dim=7 * 7 * 64, dueling_net=False, noisy_net=False):
        super(QuantileNetwork, self).__init__()
        linear = NoisyLinear if noisy_net else nn.Linear
        if not dueling_net:
            self.net = nn.Sequential(linear(embedding_dim, 512), nn.ReLU(), linear(512, num_actions))
        else:
            self.advantage_net = nn.Sequential(linear(embedding_dim, 512), nn.ReLU(), linear(512, num_actions))
            self.baseline_net = nn.Sequential(linear(embedding_dim, 512), nn.ReLU(), linear(512, 1))
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net

    def forward(self, state_embeddings, tau_embeddings):
        assert state_embeddings.shape[0] == tau_embeddings.shape[0]
        assert state_embeddings.shape[1] == tau_embeddings.shape[2]
        batch_size = state_embeddings.shape[0]
        N = tau_embeddings.shape[1]
        state_embeddings = state_embeddings.view(batch_size, 1, self.embedding_dim)
        embeddings = (state_embeddings * tau_embeddings).view(batch_size * N, self.embedding_dim)
        if not self.dueling_net:
            quantiles = self.net(embeddings)
        else:
            advantages = self.advantage_net(embeddings)
            baselines = self.baseline_net(embeddings)
            quantiles = baselines + advantages - advantages.mean(1, keepdim=True)
        return quantiles.view(batch_size, N, self.num_actions)


class IQN(BaseModel):

    def __init__(self, num_channels, num_actions, K=32, num_cosines=32, embedding_dim=7 * 7 * 64, dueling_net=False, noisy_net=False):
        super(IQN, self).__init__()
        self.dqn_net = DQNBase(num_channels=num_channels)
        self.cosine_net = CosineEmbeddingNetwork(num_cosines=num_cosines, embedding_dim=embedding_dim, noisy_net=noisy_net)
        self.quantile_net = QuantileNetwork(num_actions=num_actions, dueling_net=dueling_net, noisy_net=noisy_net)
        self.K = K
        self.num_channels = num_channels
        self.num_actions = num_actions
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net

    def calculate_state_embeddings(self, states):
        return self.dqn_net(states)

    def calculate_quantiles(self, taus, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)
        tau_embeddings = self.cosine_net(taus)
        return self.quantile_net(state_embeddings, tau_embeddings)

    def calculate_q(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None else state_embeddings.shape[0]
        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)
        taus = torch.rand(batch_size, self.K, dtype=state_embeddings.dtype, device=state_embeddings.device)
        quantiles = self.calculate_quantiles(taus, state_embeddings=state_embeddings)
        assert quantiles.shape == (batch_size, self.K, self.num_actions)
        q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)
        return q


class QRDQN(BaseModel):

    def __init__(self, num_channels, num_actions, N=200, embedding_dim=7 * 7 * 64, dueling_net=False, noisy_net=False):
        super(QRDQN, self).__init__()
        linear = NoisyLinear if noisy_net else nn.Linear
        self.dqn_net = DQNBase(num_channels=num_channels)
        if not dueling_net:
            self.q_net = nn.Sequential(linear(embedding_dim, 512), nn.ReLU(), linear(512, num_actions * N))
        else:
            self.advantage_net = nn.Sequential(linear(embedding_dim, 512), nn.ReLU(), linear(512, num_actions * N))
            self.baseline_net = nn.Sequential(linear(embedding_dim, 512), nn.ReLU(), linear(512, N))
        self.N = N
        self.num_channels = num_channels
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net

    def forward(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None else state_embeddings.shape[0]
        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)
        if not self.dueling_net:
            quantiles = self.q_net(state_embeddings).view(batch_size, self.N, self.num_actions)
        else:
            advantages = self.advantage_net(state_embeddings).view(batch_size, self.N, self.num_actions)
            baselines = self.baseline_net(state_embeddings).view(batch_size, self.N, 1)
            quantiles = baselines + advantages - advantages.mean(dim=2, keepdim=True)
        assert quantiles.shape == (batch_size, self.N, self.num_actions)
        return quantiles

    def calculate_q(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None else state_embeddings.shape[0]
        quantiles = self(states=states, state_embeddings=state_embeddings)
        q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)
        return q


def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class FractionProposalNetwork(nn.Module):

    def __init__(self, N=32, embedding_dim=7 * 7 * 64):
        super(FractionProposalNetwork, self).__init__()
        self.net = nn.Sequential(nn.Linear(embedding_dim, N)).apply(lambda x: initialize_weights_xavier(x, gain=0.01))
        self.N = N
        self.embedding_dim = embedding_dim

    def forward(self, state_embeddings):
        batch_size = state_embeddings.shape[0]
        log_probs = F.log_softmax(self.net(state_embeddings), dim=1)
        probs = log_probs.exp()
        assert probs.shape == (batch_size, self.N)
        tau_0 = torch.zeros((batch_size, 1), dtype=state_embeddings.dtype, device=state_embeddings.device)
        taus_1_N = torch.cumsum(probs, dim=1)
        taus = torch.cat((tau_0, taus_1_N), dim=1)
        assert taus.shape == (batch_size, self.N + 1)
        tau_hats = (taus[:, :-1] + taus[:, 1:]).detach() / 2.0
        assert tau_hats.shape == (batch_size, self.N)
        entropies = -(log_probs * probs).sum(dim=-1, keepdim=True)
        assert entropies.shape == (batch_size, 1)
        return taus, tau_hats, entropies


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CosineEmbeddingNetwork,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 1])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NoisyLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_toshikwa_fqf_iqn_qrdqn_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

