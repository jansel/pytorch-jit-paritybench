import sys
_module = sys.modules[__name__]
del sys
evaluate_DQN_script = _module
DQN = _module
playground = _module
train_DQN_script = _module
constants = _module
replay_buffer = _module
utils = _module
video_utils = _module

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


import matplotlib.pyplot as plt


import numpy as np


from torch import nn


import time


import copy


from torch.optim import Adam


from torch.utils.tensorboard import SummaryWriter


import random


import re


class DQN(nn.Module):
    """
    I wrote the architecture a bit more generic, hence more lines of code,
    but it's more flexible if you want to experiment with the DQN architecture.

    """

    def __init__(self, env, num_in_channels=4, number_of_actions=3, epsilon_schedule=None):
        super().__init__()
        self.env = env
        self.epsilon_schedule = epsilon_schedule
        self.num_calls_to_epsilon_greedy = 0
        num_of_filters_cnn = [num_in_channels, 32, 64, 64]
        kernel_sizes = [8, 4, 3]
        strides = [4, 2, 1]
        cnn_modules = []
        for i in range(len(num_of_filters_cnn) - 1):
            cnn_modules.extend(self._cnn_block(num_of_filters_cnn[i], num_of_filters_cnn[i + 1], kernel_sizes[i], strides[i]))
        self.cnn_part = nn.Sequential(*cnn_modules, nn.Flatten())
        with torch.no_grad():
            dummy_input = torch.from_numpy(env.observation_space.sample()[np.newaxis])
            if dummy_input.shape[1] != num_in_channels:
                assert num_in_channels % dummy_input.shape[1] == 0
                dummy_input = dummy_input.repeat(1, int(num_in_channels / dummy_input.shape[1]), 1, 1).float()
            num_nodes_fc1 = self.cnn_part(dummy_input).shape[1]
            None
        num_of_neurons_fc = [num_nodes_fc1, 512, number_of_actions]
        fc_modules = []
        for i in range(len(num_of_neurons_fc) - 1):
            last_layer = i == len(num_of_neurons_fc) - 1
            fc_modules.extend(self._fc_block(num_of_neurons_fc[i], num_of_neurons_fc[i + 1], use_relu=not last_layer))
        self.fc_part = nn.Sequential(*fc_modules)

    def forward(self, states):
        return self.fc_part(self.cnn_part(states))

    def epsilon_greedy(self, state):
        assert self.epsilon_schedule is not None, f"No schedule provided, can't call epsilon_greedy function"
        assert state.shape[0] == 1, f'Agent can only act on a single state'
        self.num_calls_to_epsilon_greedy += 1
        if np.random.rand() < self.epsilon_value():
            action = self.env.action_space.sample()
        else:
            action = self.forward(state).argmax(dim=1)[0].cpu().numpy()
        return action

    def epsilon_value(self):
        return self.epsilon_schedule(self.num_calls_to_epsilon_greedy)

    def _cnn_block(self, num_in_filters, num_out_filters, kernel_size, stride):
        layers = [nn.Conv2d(num_in_filters, num_out_filters, kernel_size=kernel_size, stride=stride), nn.ReLU()]
        return layers

    def _fc_block(self, num_in_neurons, num_out_neurons, use_relu=True):
        layers = [nn.Linear(num_in_neurons, num_out_neurons)]
        if use_relu:
            layers.append(nn.ReLU())
        return layers

