import sys
_module = sys.modules[__name__]
del sys
a3c = _module
a3c_test = _module
models = _module
utils = _module
worker = _module
Networks = _module
categoricalDQN = _module
c51 = _module
models = _module
common = _module
data_structures = _module
models = _module
replay_buffers = _module
utils = _module
wrappers = _module
clipped_ddqn = _module
ddqn = _module
duelingDQN = _module
models = _module
noisyDQN = _module
models = _module
noisy_dqn = _module
perDQN = _module
models = _module
setup = _module
dqn = _module
dueling_dqn = _module
noisy_test = _module
per_dqn = _module
noisy_test = _module
vanillaDQN = _module
dqn = _module
models = _module
Methods = _module
a2c = _module
a2c_test = _module
decoupled_a2c = _module
decoupled_a2c_test = _module
models = _module
models = _module
worker = _module
noise = _module
ddpg = _module
ddpg = _module
ddpg_test = _module
models = _module
sac = _module
models = _module
sac2018 = _module
sac2019 = _module
sac_test = _module
td3 = _module
models = _module
td3 = _module
td3_test = _module
buffer = _module
models = _module
sac2018 = _module
sac2019 = _module
models = _module
td3 = _module
agent = _module
model = _module
test = _module
agent = _module
model = _module
agent = _module
model = _module
agent = _module
model = _module
agent = _module
model = _module
reinforce = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
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


import torch.multiprocessing as mp


from torch.distributions import Categorical


import torch.autograd as autograd


import torch.optim as optim


import numpy as np


import math


import random


from collections import deque


import torch.nn


from torch.distributions import Normal


from torch.distributions import Uniform


from torch.autograd import Variable


import torch.autograd


class TwoHeadNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(TwoHeadNetwork, self).__init__()
        self.policy1 = nn.Linear(input_dim, 256)
        self.policy2 = nn.Linear(256, output_dim)
        self.value1 = nn.Linear(input_dim, 256)
        self.value2 = nn.Linear(256, 1)

    def forward(self, state):
        logits = F.relu(self.policy1(state))
        logits = self.policy2(logits)
        value = F.relu(self.value1(state))
        value = self.value2(value)
        return logits, value


class ValueNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, state):
        value = F.relu(self.fc1(state))
        value = self.fc2(value)
        return value


class PolicyNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, state):
        logits = F.relu(self.fc1(state))
        logits = self.fc2(logits)
        return logits


class DistributionalDQN(nn.Module):

    def __init__(self, input_dim, output_dim, use_conv=True, n_atoms=51,
        Vmin=-10.0, Vmax=10.0):
        super(DistributionalDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_atoms = n_atoms
        self.use_conv = use_conv
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.delta_z = (Vmax - Vmin) / (self.n_atoms - 1)
        self.support = torch.arange(self.Vmin, self.Vmax + self.delta_z,
            self.delta_z)
        self.features = self.conv_layer(self.input_dim
            ) if self.use_conv else None
        self.fc = nn.Sequential(nn.Linear(self.feature_size() if self.
            use_conv else self.input_dim[0], 128), nn.ReLU(), nn.Linear(128,
            256), nn.ReLU(), nn.Linear(256, self.output_dim * self.n_atoms))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        batch_size = state.size()[0]
        feats = conv_features(state) if self.use_conv else state
        dist = self.fc(state).view(batch_size, -1, self.n_atoms)
        probs = self.softmax(dist)
        Qvals = torch.sum(probs * self.support, dim=2)
        return dist, Qvals

    def get_q_vals(self, state):
        dist = self.forward(state)
        probs = self.softmax(dist)
        weights = probs * self.support
        qvals = weights.sum(dim=2)
        return dist, qvals

    def conv_features(self, state):
        feats = self.features(state)
        return feats.view(feats.size(0), -1)

    def conv_layer(self, input_dim):
        conv = nn.Sequential(nn.Conv2d(input_dim[0], 32, kernel_size=8,
            stride=4), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2
            ), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
            )


class VanillaDQN(nn.Module):

    def __init__(self, input_dim, output_dim, use_conv=True):
        super(NoisyDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_conv = use_conv
        self.fc_input_dim = self.input_dim[0]
        if self.use_conv:
            self.conv_net = self.get_conv_net(self.input_dim)
            self.fc_input_dim = self.feature_size()
        self.fc = nn.Sequential(nn.Linear(self.fc_input_dim, 128), nn.ReLU(
            ), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, self.output_dim))

    def forward(self, state):
        if self.use_conv:
            feats = self.conv_net(state)
            feats = feats.view(feats.size(0), -1)
        else:
            feats = state
        qvals = self.noisy_fc(feats)
        return qvals

    def get_conv_net(self, input_dim):
        conv = nn.Sequential(nn.Conv2d(input_dim[0], 32, kernel_size=8,
            stride=4), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2
            ), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
            )
        return conv

    def feature_size(self):
        return self.conv_net(autograd.Variable(torch.zeros(1, *self.input_dim))
            ).view(1, -1).size(1)


class DuelingDQN(nn.Module):

    def __init__(self, input_dim, output_dim, use_conv=True, noisy=True):
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_conv = use_conv
        self.features = self.conv_layer(self.input_dim
            ) if self.use_conv else nn.Sequential(nn.Linear(self.input_dim[
            0], 128), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(
            ), nn.Linear(128, self.output_dim))

    def forward(self, state):
        feats = self.conv_features(state) if self.use_conv else self.features(
            state)
        values = self.value_stream(feats)
        advantages = self.advantage_stream(feats)
        qvals = values + (advantages - advantages.mean())
        return qvals

    def conv_features(self, state):
        feats = self.features(state)
        return feats.view(feats.size(0), -1)

    def conv_layer(self, input_dim):
        conv = nn.Sequential(nn.Conv2d(input_dim[0], 32, kernel_size=8,
            stride=4), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2
            ), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
            )
        return conv

    def feature_size(self, input_dim):
        return self.features(autograd.Variable(torch.zeros(1, *input_dim))
            ).view(1, -1).size(1)


class DistributionalDQN(nn.Module):

    def __init__(self, input_dim, output_dim, use_conv=True, n_atoms=51,
        Vmin=-10.0, Vmax=10.0):
        super(DistributionalDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_atoms = n_atoms
        self.use_conv = use_conv
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.delta_z = (Vmax - Vmin) / (self.n_atoms - 1)
        self.support = torch.arange(self.Vmin, self.Vmax + self.delta_z,
            self.delta_z)
        self.features = self.conv_layer(self.input_dim
            ) if self.use_conv else None
        self.fc = nn.Sequential(nn.Linear(self.feature_size() if self.
            use_conv else self.input_dim[0], 128), nn.ReLU(), nn.Linear(128,
            256), nn.ReLU(), nn.Linear(256, self.output_dim * self.n_atoms))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        batch_size = state.size()[0]
        feats = conv_features(state) if self.use_conv else state
        dist = self.fc(state).view(batch_size, -1, self.n_atoms)
        probs = self.softmax(dist)
        Qvals = torch.sum(probs * self.support, dim=2)
        return dist, Qvals

    def get_q_vals(self, state):
        dist = self.forward(state)
        probs = self.softmax(dist)
        weights = probs * self.support
        qvals = weights.sum(dim=2)
        return dist, qvals

    def conv_features(self, state):
        feats = self.features(state)
        return feats.view(feats.size(0), -1)

    def conv_layer(self, input_dim):
        conv = nn.Sequential(nn.Conv2d(input_dim[0], 32, kernel_size=8,
            stride=4), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2
            ), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
            )


class RecurrentDQN(nn.Module):

    def __init__(self, input_dim, gru_size, output_dim, use_conv=True):
        super(RecurrentDQN, self).__init__()
        self.input_dim = input_dim
        self.gru_size = gru_size
        self.output_dim = output_dim
        self.use_conv = use_conv
        self.features = self.conv_layer() if self.use_conv else None
        self.linear1 = nn.Linear(self.feature_size() if self.use_conv else
            self.input_dim[0], self.gru_size)
        self.gru = nn.GRUCell(self.gru_size, self.gru_size)
        self.linear2 = nn.Linear(self.gru_size, self.output_dim)

    def forward(self, state_input, hidden_state):
        feats = self.conv_features(state_input
            ) if self.use_conv else state_input
        x = F.relu(self.linear1(feats))
        h_in = hidden_state.reshape(-1, self.gru_size)
        h = self.gru(x, h_in)
        q = self.linear2(h)
        return q, h

    def init_hidden(self):
        return self.linear1.weight.new(1, self.gru_size).zero_()

    def conv_features(self, state):
        feats = self.features(state)
        return feats.view(feats.size(0), -1)

    def conv_layer(self, input_dim):
        conv = nn.Sequential(nn.Conv2d(input_dim[0], 32, kernel_size=8,
            stride=4), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2
            ), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
            )
        return conv

    def feature_size(self, input_dim):
        return self.features(autograd.Variable(torch.zeros(1, *input_dim))
            ).view(1, -1).size(1)


class NoisyDQN(nn.Module):

    def __init__(self, input_dim, output_dim, use_conv=True, noisy=True):
        super(NoisyDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_conv = use_conv
        self.noisy = noisy
        self.fc_input_dim = self.input_dim[0]
        if self.use_conv:
            self.conv_net = self.get_conv_net(self.input_dim)
            self.fc_input_dim = self.feature_size()
        if self.noisy:
            self.noisy_fc = nn.Sequential(NoisyLinear(self.fc_input_dim, 
                512), nn.ReLU(), NoisyLinear(512, self.output_dim))
        if not self.noisy:
            self.fc = nn.Sequential(nn.Linear(self.fc_input_dim, 128), nn.
                ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, self
                .output_dim))

    def forward(self, state):
        feats = self.conv_net(state)
        feats = feats.view(feats.size(0), -1)
        qvals = self.noisy_fc(feats)
        return qvals

    def get_conv_net(self, input_dim):
        conv = nn.Sequential(nn.Conv2d(input_dim[0], 32, kernel_size=8,
            stride=4), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2
            ), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
            )
        return conv

    def feature_size(self):
        return self.conv_net(autograd.Variable(torch.zeros(1, *self.input_dim))
            ).view(1, -1).size(1)


class NoisyLinear(nn.Module):

    def __init__(self, num_in, num_out, is_training=True):
        super(NoisyLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.is_training = is_training
        self.mu_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.mu_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.sigma_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.register_buffer('epsilon_weight', torch.FloatTensor(num_out,
            num_in))
        self.register_buffer('epsilon_bias', torch.FloatTensor(num_out))
        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        self.reset_noise()
        if self.is_training:
            weight = self.mu_weight + self.sigma_weight.mul(autograd.
                Variable(self.epsilon_weight))
            bias = self.mu_bias + self.sigma_bias.mul(autograd.Variable(
                self.epsilon_bias))
        else:
            weight = self.mu_weight
            buas = self.mu_bias
        y = F.linear(x, weight, bias)
        return y

    def reset_parameters(self):
        std = math.sqrt(3 / self.num_in)
        self.mu_weight.data.uniform_(-std, std)
        self.mu_bias.data.uniform_(-std, std)
        self.sigma_weight.data.fill_(0.017)
        self.sigma_bias.data.fill_(0.017)

    def reset_noise(self):
        self.epsilon_weight.data.normal_()
        self.epsilon_bias.data.normal_()


class FactorizedNoisyLinear(nn.Module):

    def __init__(self, num_in, num_out, is_training=True):
        super(FactorizedNoisyLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.is_training = is_training
        self.mu_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.mu_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.sigma_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.register_buffer('epsilon_i', torch.FloatTensor(num_in))
        self.register_buffer('epsilon_j', torch.FloatTensor(num_out))
        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        self.reset_noise()
        if self.is_training:
            epsilon_weight = self.epsilon_j.ger(self.epsilon_i)
            epsilon_bias = self.epsilon_j
            weight = self.mu_weight + self.sigma_weight.mul(autograd.
                Variable(epsilon_weight))
            bias = self.mu_bias + self.sigma_bias.mul(autograd.Variable(
                epsilon_bias))
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        y = F.linear(x, weight, bias)
        return y

    def reset_parameters(self):
        std = 1 / math.sqrt(self.num_in)
        self.mu_weight.data.uniform_(-std, std)
        self.mu_bias.data.uniform_(-std, std)
        self.sigma_weight.data.fill_(0.5 / math.sqrt(self.num_in))
        self.sigma_bias.data.fill_(0.5 / math.sqrt(self.num_in))

    def reset_noise(self):
        eps_i = torch.randn(self.num_in)
        eps_j = torch.randn(self.num_out)
        self.epsilon_i = eps_i.sign() * eps_i.abs().sqrt()
        self.epsilon_j = eps_j.sign() * eps_j.abs().sqrt()


class ConvDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ConvDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_input_dim = self.feature_size()
        self.conv = nn.Sequential(nn.Conv2d(self.input_dim[0], 32,
            kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64,
            kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64,
            kernel_size=3, stride=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(self.fc_input_dim, 128), nn.ReLU(
            ), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, self.output_dim))

    def forward(self, state):
        features = self.conv_net(state)
        features = features.view(features.size(0), -1)
        qvals = self.fc(features)
        return qvals

    def feature_size(self):
        return self.conv_net(autograd.Variable(torch.zeros(1, *self.input_dim))
            ).view(1, -1).size(1)


class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential(nn.Linear(self.input_dim[0], 128), nn.ReLU(
            ), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, self.output_dim))

    def forward(self, state):
        qvals = self.fc(state)
        return qvals


class ConvDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ConvDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_input_dim = self.feature_size()
        self.conv = nn.Sequential(nn.Conv2d(self.input_dim[0], 32,
            kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64,
            kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64,
            kernel_size=3, stride=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(self.fc_input_dim, 128), nn.ReLU(
            ), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, self.output_dim))

    def forward(self, state):
        features = self.conv_net(state)
        features = features.view(features.size(0), -1)
        qvals = self.fc(features)
        return qvals

    def feature_size(self):
        return self.conv_net(autograd.Variable(torch.zeros(1, *self.input_dim))
            ).view(1, -1).size(1)


class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential(nn.Linear(self.input_dim[0], 128), nn.ReLU(
            ), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, self.output_dim))

    def forward(self, state):
        qvals = self.fc(state)
        return qvals


class ConvDuelingDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ConvDuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_input_dim = self.feature_size()
        self.conv = nn.Sequential(nn.Conv2d(input_dim[0], 32, kernel_size=8,
            stride=4), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2
            ), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
            )
        self.value_stream = nn.Sequential(nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(), nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(self.fc_input_dim, 
            128), nn.ReLU(), nn.Linear(128, self.output_dim))

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        return qvals

    def feature_size(self):
        return self.conv(autograd.Variable(torch.zeros(1, *self.input_dim))
            ).view(1, -1).size(1)


class DuelingDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.feauture_layer = nn.Sequential(nn.Linear(self.input_dim[0], 
            128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(
            ), nn.Linear(128, self.output_dim))

    def forward(self, state):
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        return qvals


class ConvNoisyDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ConvNoisyDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = nn.Sequential(nn.Conv2d(self.input_dim[0], 32,
            kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64,
            kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64,
            kernel_size=3, stride=1), nn.ReLU())
        self.noisy_fc = nn.Sequential(FactorizedNoisyLinear(self.
            feature_size(), 512), nn.ReLU(), FactorizedNoisyLinear(512,
            self.output_dim))

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        qvals = self.noisy_fc(features)
        return qvals

    def feature_size(self):
        return self.conv(autograd.Variable(torch.zeros(1, *self.input_dim))
            ).view(1, -1).size(1)


class NoisyDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(NoisyDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noisy_fc = nn.Sequential(nn.Linear(self.input_dim[0], 128), nn
            .ReLU(), FactorizedNoisyLinear(128, 128), nn.ReLU(),
            FactorizedNoisyLinear(128, self.output_dim))

    def forward(self, state):
        qvals = self.noisy_fc(state)
        return qvals


class ConvDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ConvDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_input_dim = self.feature_size()
        self.conv = nn.Sequential(nn.Conv2d(self.input_dim[0], 32,
            kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64,
            kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64,
            kernel_size=3, stride=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(self.fc_input_dim, 128), nn.ReLU(
            ), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, self.output_dim))

    def forward(self, state):
        features = self.conv_net(state)
        features = features.view(features.size(0), -1)
        qvals = self.fc(features)
        return qvals

    def feature_size(self):
        return self.conv_net(autograd.Variable(torch.zeros(1, *self.input_dim))
            ).view(1, -1).size(1)


class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential(nn.Linear(self.input_dim[0], 128), nn.ReLU(
            ), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, self.output_dim))

    def forward(self, state):
        qvals = self.fc(state)
        return qvals


class ConvDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ConvDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_input_dim = self.feature_size()
        self.conv = nn.Sequential(nn.Conv2d(self.input_dim[0], 32,
            kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64,
            kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64,
            kernel_size=3, stride=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(self.fc_input_dim, 128), nn.ReLU(
            ), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, self.output_dim))

    def forward(self, state):
        features = self.conv_net(state)
        features = features.view(features.size(0), -1)
        qvals = self.fc(features)
        return qvals

    def feature_size(self):
        return self.conv_net(autograd.Variable(torch.zeros(1, *self.input_dim))
            ).view(1, -1).size(1)


class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential(nn.Linear(self.input_dim[0], 128), nn.ReLU(
            ), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, self.output_dim))

    def forward(self, state):
        qvals = self.fc(state)
        return qvals


class TwoHeadNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(TwoHeadNetwork, self).__init__()
        self.policy1 = nn.Linear(input_dim, 256)
        self.policy2 = nn.Linear(256, output_dim)
        self.value1 = nn.Linear(input_dim, 256)
        self.value2 = nn.Linear(256, 1)

    def forward(self, state):
        logits = F.relu(self.policy1(state))
        logits = self.policy2(logits)
        value = F.relu(self.value1(state))
        value = self.value2(value)
        return logits, value


class ValueNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, state):
        value = F.relu(self.fc1(state))
        value = self.fc2(value)
        return value


class PolicyNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, state):
        logits = F.relu(self.fc1(state))
        logits = self.fc2(logits)
        return logits


class TwoHeadNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(TwoHeadNetwork, self).__init__()
        self.policy1 = nn.Linear(input_dim, 256)
        self.policy2 = nn.Linear(256, output_dim)
        self.value1 = nn.Linear(input_dim, 256)
        self.value2 = nn.Linear(256, 1)

    def forward(self, state):
        logits = F.relu(self.policy1(state))
        logits = self.policy2(logits)
        value = F.relu(self.value1(state))
        value = self.value2(value)
        return logits, value


class ValueNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, state):
        value = F.relu(self.fc1(state))
        value = self.fc2(value)
        return value


class PolicyNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, state):
        logits = F.relu(self.fc1(state))
        logits = self.fc2(logits)
        return logits


class NoisyLinear(nn.Module):

    def __init__(self, num_in, num_out, is_training=True):
        super(NoisyLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.is_training = is_training
        self.mu_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.mu_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.sigma_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.register_buffer('epsilon_weight', torch.FloatTensor(num_out,
            num_in))
        self.register_buffer('epsilon_bias', torch.FloatTensor(num_out))
        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        self.reset_noise()
        if self.is_training:
            weight = self.mu_weight + self.sigma_weight.mul(autograd.
                Variable(self.epsilon_weight))
            bias = self.mu_bias + self.sigma_bias.mul(autograd.Variable(
                self.epsilon_bias))
        else:
            weight = self.mu_weight
            buas = self.mu_bias
        y = F.linear(x, weight, bias)
        return y

    def reset_parameters(self):
        std = math.sqrt(3 / self.num_in)
        self.mu_weight.data.uniform_(-std, std)
        self.mu_bias.data.uniform_(-std, std)
        self.sigma_weight.data.fill_(0.017)
        self.sigma_bias.data.fill_(0.017)

    def reset_noise(self):
        self.epsilon_weight.data.normal_()
        self.epsilon_bias.data.normal_()


class FactorizedNoisyLinear(nn.Module):

    def __init__(self, num_in, num_out, is_training=True):
        super(FactorizedNoisyLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.is_training = is_training
        self.mu_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.mu_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.sigma_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.register_buffer('epsilon_i', torch.FloatTensor(num_in))
        self.register_buffer('epsilon_j', torch.FloatTensor(num_out))
        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        self.reset_noise()
        if self.is_training:
            epsilon_weight = self.epsilon_j.ger(self.epsilon_i)
            epsilon_bias = self.epsilon_j
            weight = self.mu_weight + self.sigma_weight.mul(autograd.
                Variable(epsilon_weight))
            bias = self.mu_bias + self.sigma_bias.mul(autograd.Variable(
                epsilon_bias))
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        y = F.linear(x, weight, bias)
        return y

    def reset_parameters(self):
        std = 1 / math.sqrt(self.num_in)
        self.mu_weight.data.uniform_(-std, std)
        self.mu_bias.data.uniform_(-std, std)
        self.sigma_weight.data.fill_(0.5 / math.sqrt(self.num_in))
        self.sigma_bias.data.fill_(0.5 / math.sqrt(self.num_in))

    def reset_noise(self):
        eps_i = torch.randn(self.num_in)
        eps_j = torch.randn(self.num_out)
        self.epsilon_i = eps_i.sign() * eps_i.abs().sqrt()
        self.epsilon_j = eps_j.sign() * eps_j.abs().sqrt()


class Critic(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.linear1 = nn.Linear(self.obs_dim, 1024)
        self.linear2 = nn.Linear(1024 + self.action_dim, 512)
        self.linear3 = nn.Linear(512, 300)
        self.linear4 = nn.Linear(300, 1)

    def forward(self, x, a):
        x = F.relu(self.linear1(x))
        xa_cat = torch.cat([x, a], 1)
        xa = F.relu(self.linear2(xa_cat))
        xa = F.relu(self.linear3(xa))
        qval = self.linear4(xa)
        return qval


class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.linear1 = nn.Linear(self.obs_dim, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, self.action_dim)

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class ValueNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, init_w=0.003):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SoftQNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=0.003):
        super(SoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=
        0.003, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, epsilon=1e-06):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_pi = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)
        return action, log_pi


class Critic(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.linear1 = nn.Linear(self.obs_dim, 1024)
        self.linear2 = nn.Linear(1024 + self.action_dim, 512)
        self.linear3 = nn.Linear(512, 300)
        self.linear4 = nn.Linear(300, 1)

    def forward(self, x, a):
        x = F.relu(self.linear1(x))
        xa_cat = torch.cat([x, a], 1)
        xa = F.relu(self.linear2(xa_cat))
        xa = F.relu(self.linear3(xa))
        qval = self.linear4(xa)
        return qval


class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.linear1 = nn.Linear(self.obs_dim, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, self.action_dim)

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class ValueNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, init_w=0.003):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SoftQNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=0.003):
        super(SoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class GaussianPolicy(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=
        0.003, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, epsilon=1e-06):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        log_pi = (normal.log_prob(z) - torch.log(1 - torch.tanh(z).pow(2) +
            epsilon)).sum(1, keepdim=True)
        return mean, std, z, log_pi


class Critic(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.linear1 = nn.Linear(self.obs_dim, 1024)
        self.linear2 = nn.Linear(1024 + self.action_dim, 512)
        self.linear3 = nn.Linear(512, 300)
        self.linear4 = nn.Linear(300, 1)

    def forward(self, x, a):
        x = F.relu(self.linear1(x))
        xa_cat = torch.cat([x, a], 1)
        xa = F.relu(self.linear2(xa_cat))
        xa = F.relu(self.linear3(xa))
        qval = self.linear4(xa)
        return qval


class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.linear1 = nn.Linear(self.obs_dim, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, self.action_dim)

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class ActorCritic(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size=256,
        learning_rate=0.0003):
        super(ActorCritic, self).__init__()
        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)
        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state_tensor):
        value = F.relu(self.critic_linear1(state_tensor))
        value = self.critic_linear2(value)
        policy_dist = F.relu(self.actor_linear1(state_tensor))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)
        return value, policy_dist


class Critic(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Actor(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, learning_rate=
        0.0003):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class DQN(nn.Module):

    def __init__(self, num_in, num_out):
        super(DQN, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.linear1 = nn.Linear(self.num_in, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, self.num_out)

    def forward(self, state_tensor):
        qvals = F.relu(self.linear1(state_tensor))
        qvals = F.relu(self.linear2(qvals))
        qvals = self.linear3(qvals)
        return qvals


class CnnDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(CnnDQN, self).__init__()
        self.input_dim = input_dim
        self.action_space_dim = output_dim
        self.features = nn.Sequential(nn.Conv2d(input_dim[0], 32,
            kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64,
            kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64,
            kernel_size=3, stride=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(self.feature_size(), 512), nn.
            ReLU(), nn.Linear(512, self.action_space_dim))

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_dim))
            ).view(1, -1).size(1)

    def forward(self, state):
        qvals = self.features(state)
        qvals = qvals.view(qvals.size(0), -1)
        qvals = self.fc(qvals)
        return qvals


class DQN(nn.Module):

    def __init__(self, num_in, num_out):
        super(DQN, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.linear1 = nn.Linear(self.num_in, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, self.num_out)

    def forward(self, state_tensor):
        qvals = F.relu(self.linear1(state_tensor))
        qvals = F.relu(self.linear2(qvals))
        qvals = self.linear3(qvals)
        return qvals


class CnnDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(CnnDQN, self).__init__()
        self.input_dim = input_dim
        self.action_space_dim = output_dim
        self.features = nn.Sequential(nn.Conv2d(input_dim[0], 32,
            kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64,
            kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64,
            kernel_size=3, stride=1), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(self.feature_size(), 512), nn.
            ReLU(), nn.Linear(512, self.action_space_dim))

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_dim))
            ).view(1, -1).size(1)

    def forward(self, state):
        qvals = self.features(state)
        qvals = qvals.view(qvals.size(0), -1)
        qvals = self.fc(qvals)
        return qvals


class DDQN(nn.Module):

    def __init__(self, num_in, num_out):
        super(DDQN, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.features = nn.Sequential(nn.Linear(self.num_in, 128), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(
            ), nn.Linear(128, self.num_out))

    def forward(self, state_tensor):
        x = self.features(state_tensor)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        qvals = value + (advantage - advantage.mean())
        return qvals


class CnnDDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(CnnDDQN, self).__init__()
        self.num_in = input_dim
        self.num_out = output_dim
        self.features = nn.Sequential(nn.Conv2d(self.num_in[0], 32,
            kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64,
            kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64,
            kernel_size=3, stride=1), nn.ReLU())
        self.fc_value = nn.Sequential(nn.Linear(self.feature_size(), 512),
            nn.ReLU(), nn.Linear(512, self.num_out))
        self.fc_advantage = nn.Sequential(nn.Linear(self.feature_size(), 
            512), nn.ReLU(), nn.Linear(512, self.num_out))

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_dim))
            ).view(1, -1).size(1)

    def forward(self, state_tensor):
        x = self.features(state_tensor)
        x = x.view(x.size(0), -1)
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        qvals = value + (advantage - advantage.mean())
        return qvals


class PolicyNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(PolicyNetwork, self).__init__()
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_cyoon1729_Reinforcement_learning(_paritybench_base):
    pass
    def test_000(self):
        self._check(Actor(*[], **{'input_size': 4, 'hidden_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(ActorCritic(*[], **{'num_inputs': 4, 'num_actions': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Critic(*[], **{'input_size': 4, 'hidden_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(DDQN(*[], **{'num_in': 4, 'num_out': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(DQN(*[], **{'num_in': 4, 'num_out': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(DuelingDQN(*[], **{'input_dim': [4, 4], 'output_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(FactorizedNoisyLinear(*[], **{'num_in': 4, 'num_out': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(GaussianPolicy(*[], **{'num_inputs': 4, 'num_actions': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(NoisyDQN(*[], **{'input_dim': [4, 4], 'output_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(NoisyLinear(*[], **{'num_in': 4, 'num_out': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_010(self):
        self._check(PolicyNetwork(*[], **{'num_inputs': 4, 'num_actions': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_011(self):
        self._check(SoftQNetwork(*[], **{'num_inputs': 4, 'num_actions': 4}), [torch.rand([4, 4, 4, 8]), torch.rand([4, 4, 4, 8])], {})

    def test_012(self):
        self._check(TwoHeadNetwork(*[], **{'input_dim': 4, 'output_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(ValueNetwork(*[], **{'input_dim': 4, 'output_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

