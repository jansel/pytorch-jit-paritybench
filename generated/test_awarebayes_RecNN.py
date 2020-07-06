import sys
_module = sys.modules[__name__]
del sys
learning = _module
main = _module
conf = _module
streamlit_demo = _module
recnn = _module
data = _module
dataset_functions = _module
env = _module
utils = _module
nn = _module
algo = _module
models = _module
update = _module
bcq = _module
ddpg = _module
misc = _module
reinforce = _module
td3 = _module
optim = _module
TCN = _module
rep = _module
plot = _module
setup = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import numpy as np


import copy


import random


from scipy.spatial import distance


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from sklearn.model_selection import train_test_split


import torch.nn as nn


import torch.nn.functional as F


from torch.distributions import Categorical


import torch.functional as F


import math


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


import itertools as it


from torch.nn.utils import weight_norm


from scipy import ndimage


from scipy import stats


class AnomalyDetector(nn.Module):
    """
    Anomaly detector used for debugging. Basically an auto encoder.
    P.S. You need to use different weights for different embeddings.
    """

    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.ae = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Linear(32, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Linear(64, 128), nn.ReLU())

    def forward(self, x):
        """"""
        return self.ae(x)

    def rec_error(self, x):
        error = torch.sum((x - self.ae(x)) ** 2, 1)
        if x.size(1) != 1:
            return error.detach()
        return error.item()


class Actor(nn.Module):
    """
    Vanilla actor. Takes state as an argument, returns action.
    """

    def __init__(self, input_dim, action_dim, hidden_size, init_w=0.2):
        super(Actor, self).__init__()
        self.drop_layer = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_dim)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, tanh=False):
        """
        :param action: nothing should be provided here.
        :param state: state
        :param tanh: whether to use tahn as action activation
        :return: action
        """
        action = F.relu(self.linear1(state))
        action = self.drop_layer(action)
        action = F.relu(self.linear2(action))
        action = self.drop_layer(action)
        action = self.linear3(action)
        if tanh:
            action = F.tanh(action)
        return action


class DiscreteActor(nn.Module):

    def __init__(self, input_dim, action_dim, hidden_size, init_w=0):
        super(DiscreteActor, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_dim)
        self.saved_log_probs = []
        self.rewards = []
        self.correction = []
        self.lambda_k = []
        self.action_source = {'pi': 'pi', 'beta': 'beta'}
        self.select_action = self._select_action

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        action_scores = self.linear2(x)
        return F.softmax(action_scores)

    def gc(self):
        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.correction[:]
        del self.lambda_k[:]

    def _select_action(self, state, **kwargs):
        pi_probs = self.forward(state)
        pi_categorical = Categorical(pi_probs)
        pi_action = pi_categorical.sample()
        self.saved_log_probs.append(pi_categorical.log_prob(pi_action))
        return pi_probs

    def pi_beta_sample(self, state, beta, action, **kwargs):
        beta_probs = beta(state.detach(), action=action)
        pi_probs = self.forward(state)
        beta_categorical = Categorical(beta_probs)
        pi_categorical = Categorical(pi_probs)
        available_actions = {'pi': pi_categorical.sample(), 'beta': beta_categorical.sample()}
        pi_action = available_actions[self.action_source['pi']]
        beta_action = available_actions[self.action_source['beta']]
        pi_log_prob = pi_categorical.log_prob(pi_action)
        beta_log_prob = beta_categorical.log_prob(beta_action)
        return pi_log_prob, beta_log_prob, pi_probs

    def _select_action_with_correction(self, state, beta, action, writer, step, **kwargs):
        pi_log_prob, beta_log_prob, pi_probs = self.pi_beta_sample(state, beta, action)
        corr = torch.exp(pi_log_prob) / torch.exp(beta_log_prob)
        writer.add_histogram('correction', corr, step)
        writer.add_histogram('pi_log_prob', pi_log_prob, step)
        writer.add_histogram('beta_log_prob', beta_log_prob, step)
        self.correction.append(corr)
        self.saved_log_probs.append(pi_log_prob)
        return pi_probs

    def _select_action_with_TopK_correction(self, state, beta, action, K, writer, step, **kwargs):
        pi_log_prob, beta_log_prob, pi_probs = self.pi_beta_sample(state, beta, action)
        corr = torch.exp(pi_log_prob) / torch.exp(beta_log_prob)
        l_k = K * (1 - torch.exp(pi_log_prob)) ** (K - 1)
        writer.add_histogram('correction', corr, step)
        writer.add_histogram('l_k', l_k, step)
        writer.add_histogram('pi_log_prob', pi_log_prob, step)
        writer.add_histogram('beta_log_prob', beta_log_prob, step)
        self.correction.append(corr)
        self.lambda_k.append(l_k)
        self.saved_log_probs.append(pi_log_prob)
        return pi_probs


class Critic(nn.Module):
    """
    Vanilla critic. Takes state and action as an argument, returns value.
    """

    def __init__(self, input_dim, action_dim, hidden_size, init_w=3e-05):
        super(Critic, self).__init__()
        self.drop_layer = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(input_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        """"""
        value = torch.cat([state, action], 1)
        value = F.relu(self.linear1(value))
        value = self.drop_layer(value)
        value = F.relu(self.linear2(value))
        value = self.drop_layer(value)
        value = self.linear3(value)
        return value


class bcqPerturbator(nn.Module):
    """
    Batch constrained perturbative actor. Takes action as an argument, adjusts it.
    """

    def __init__(self, num_inputs, num_actions, hidden_size, init_w=0.3):
        super(bcqPerturbator, self).__init__()
        self.drop_layer = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        """"""
        a = torch.cat([state, action], 1)
        a = F.relu(self.linear1(a))
        a = self.drop_layer(a)
        a = F.relu(self.linear2(a))
        a = self.drop_layer(a)
        a = self.linear3(a)
        return a + action


class bcqGenerator(nn.Module):
    """
    Batch constrained generator. Basically VAE
    """

    def __init__(self, state_dim, action_dim, latent_dim):
        super(bcqGenerator, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)
        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)
        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)
        self.latent_dim = latent_dim
        self.normal = torch.distributions.Normal(0, 1)

    def forward(self, state, action):
        """"""
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))
        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * self.normal.sample(std.size())
        u = self.decode(state, z)
        return u, mean, std

    def decode(self, state, z=None):
        if z is None:
            z = self.normal.sample([state.size(0), self.latent_dim])
            z = z.clamp(-0.5, 0.5)
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.d3(a)


class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Actor,
     lambda: ([], {'input_dim': 4, 'action_dim': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (AnomalyDetector,
     lambda: ([], {}),
     lambda: ([torch.rand([128, 128])], {}),
     True),
    (Chomp1d,
     lambda: ([], {'chomp_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Critic,
     lambda: ([], {'input_dim': 4, 'action_dim': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 8]), torch.rand([4, 4, 4, 8])], {}),
     True),
    (DiscreteActor,
     lambda: ([], {'input_dim': 4, 'action_dim': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TemporalConvNet,
     lambda: ([], {'num_inputs': 4, 'num_channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
    (bcqGenerator,
     lambda: ([], {'state_dim': 4, 'action_dim': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (bcqPerturbator,
     lambda: ([], {'num_inputs': 4, 'num_actions': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
]

class Test_awarebayes_RecNN(_paritybench_base):
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

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

