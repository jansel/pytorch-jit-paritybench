import sys
_module = sys.modules[__name__]
del sys
diagnose_model = _module
abstract_game = _module
atari = _module
breakout = _module
cartpole = _module
connect4 = _module
gomoku = _module
gridworld = _module
lunarlander = _module
simple_grid = _module
spiel = _module
tictactoe = _module
twentyone = _module
models = _module
muzero = _module
replay_buffer = _module
self_play = _module
shared_storage = _module
trainer = _module

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


import matplotlib.pyplot as plt


import numpy


import torch


import math


import numpy as np


from abc import ABC


from abc import abstractmethod


import copy


import time


from torch.utils.tensorboard import SummaryWriter


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class AbstractNetwork(ABC, torch.nn.Module):

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


def mlp(input_size, layer_sizes, output_size, output_activation=torch.nn.Identity, activation=torch.nn.ELU):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)


class MuZeroFullyConnectedNetwork(AbstractNetwork):

    def __init__(self, observation_shape, stacked_observations, action_space_size, encoding_size, fc_reward_layers, fc_value_layers, fc_policy_layers, fc_representation_layers, fc_dynamics_layers, support_size):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        self.representation_network = torch.nn.DataParallel(mlp(observation_shape[0] * observation_shape[1] * observation_shape[2] * (stacked_observations + 1) + stacked_observations * observation_shape[1] * observation_shape[2], fc_representation_layers, encoding_size))
        self.dynamics_encoded_state_network = torch.nn.DataParallel(mlp(encoding_size + self.action_space_size, fc_dynamics_layers, encoding_size))
        self.dynamics_reward_network = torch.nn.DataParallel(mlp(encoding_size, fc_reward_layers, self.full_support_size))
        self.prediction_policy_network = torch.nn.DataParallel(mlp(encoding_size, fc_policy_layers, self.action_space_size))
        self.prediction_value_network = torch.nn.DataParallel(mlp(encoding_size, fc_value_layers, self.full_support_size))

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation.view(observation.shape[0], -1))
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-05] += 1e-05
        encoded_state_normalized = (encoded_state - min_encoded_state) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        action_one_hot = torch.zeros((action.shape[0], self.action_space_size)).float()
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state = self.dynamics_encoded_state_network(x)
        reward = self.dynamics_reward_network(next_encoded_state)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-05] += 1e-05
        next_encoded_state_normalized = (next_encoded_state - min_next_encoded_state) / scale_next_encoded_state
        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        reward = torch.log(torch.zeros(1, self.full_support_size).scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0).repeat(len(observation), 1))
        return value, reward, policy_logits, encoded_state

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(torch.nn.Module):

    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = torch.nn.functional.relu(out)
        return out


class DownSample(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.resblocks1 = torch.nn.ModuleList([ResidualBlock(out_channels // 2) for _ in range(2)])
        self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.resblocks2 = torch.nn.ModuleList([ResidualBlock(out_channels) for _ in range(3)])
        self.pooling1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = torch.nn.ModuleList([ResidualBlock(out_channels) for _ in range(3)])
        self.pooling2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.conv2(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


class DownsampleCNN(torch.nn.Module):

    def __init__(self, in_channels, out_channels, h_w):
        super().__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.features = torch.nn.Sequential(torch.nn.Conv2d(in_channels, mid_channels, kernel_size=h_w[0] * 2, stride=4, padding=2), torch.nn.ReLU(inplace=True), torch.nn.MaxPool2d(kernel_size=3, stride=2), torch.nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2), torch.nn.ReLU(inplace=True), torch.nn.MaxPool2d(kernel_size=3, stride=2))
        self.avgpool = torch.nn.AdaptiveAvgPool2d(h_w)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x


class RepresentationNetwork(torch.nn.Module):

    def __init__(self, observation_shape, stacked_observations, num_blocks, num_channels, downsample):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            if self.downsample == 'resnet':
                self.downsample_net = DownSample(observation_shape[0] * (stacked_observations + 1) + stacked_observations, num_channels)
            elif self.downsample == 'CNN':
                self.downsample_net = DownsampleCNN(observation_shape[0] * (stacked_observations + 1) + stacked_observations, num_channels, (math.ceil(observation_shape[1] / 16), math.ceil(observation_shape[2] / 16)))
            else:
                raise NotImplementedError('downsample should be "resnet" or "CNN".')
        self.conv = conv3x3(observation_shape[0] * (stacked_observations + 1) + stacked_observations, num_channels)
        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.resblocks = torch.nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_blocks)])

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = torch.nn.functional.relu(x)
        for block in self.resblocks:
            x = block(x)
        return x


class DynamicsNetwork(torch.nn.Module):

    def __init__(self, num_blocks, num_channels, reduced_channels_reward, fc_reward_layers, full_support_size, block_output_size_reward):
        super().__init__()
        self.conv = conv3x3(num_channels, num_channels - 1)
        self.bn = torch.nn.BatchNorm2d(num_channels - 1)
        self.resblocks = torch.nn.ModuleList([ResidualBlock(num_channels - 1) for _ in range(num_blocks)])
        self.conv1x1_reward = torch.nn.Conv2d(num_channels - 1, reduced_channels_reward, 1)
        self.block_output_size_reward = block_output_size_reward
        self.fc = mlp(self.block_output_size_reward, fc_reward_layers, full_support_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.relu(x)
        for block in self.resblocks:
            x = block(x)
        state = x
        x = self.conv1x1_reward(x)
        x = x.view(-1, self.block_output_size_reward)
        reward = self.fc(x)
        return state, reward


class PredictionNetwork(torch.nn.Module):

    def __init__(self, action_space_size, num_blocks, num_channels, reduced_channels_value, reduced_channels_policy, fc_value_layers, fc_policy_layers, full_support_size, block_output_size_value, block_output_size_policy):
        super().__init__()
        self.resblocks = torch.nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.conv1x1_value = torch.nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = torch.nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = mlp(self.block_output_size_value, fc_value_layers, full_support_size)
        self.fc_policy = mlp(self.block_output_size_policy, fc_policy_layers, action_space_size)

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1_value(x)
        policy = self.conv1x1_policy(x)
        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


class MuZeroResidualNetwork(AbstractNetwork):

    def __init__(self, observation_shape, stacked_observations, action_space_size, num_blocks, num_channels, reduced_channels_reward, reduced_channels_value, reduced_channels_policy, fc_reward_layers, fc_value_layers, fc_policy_layers, support_size, downsample):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        block_output_size_reward = reduced_channels_reward * math.ceil(observation_shape[1] / 16) * math.ceil(observation_shape[2] / 16) if downsample else reduced_channels_reward * observation_shape[1] * observation_shape[2]
        block_output_size_value = reduced_channels_value * math.ceil(observation_shape[1] / 16) * math.ceil(observation_shape[2] / 16) if downsample else reduced_channels_value * observation_shape[1] * observation_shape[2]
        block_output_size_policy = reduced_channels_policy * math.ceil(observation_shape[1] / 16) * math.ceil(observation_shape[2] / 16) if downsample else reduced_channels_policy * observation_shape[1] * observation_shape[2]
        self.representation_network = torch.nn.DataParallel(RepresentationNetwork(observation_shape, stacked_observations, num_blocks, num_channels, downsample))
        self.dynamics_network = torch.nn.DataParallel(DynamicsNetwork(num_blocks, num_channels + 1, reduced_channels_reward, fc_reward_layers, self.full_support_size, block_output_size_reward))
        self.prediction_network = torch.nn.DataParallel(PredictionNetwork(action_space_size, num_blocks, num_channels, reduced_channels_value, reduced_channels_policy, fc_value_layers, fc_policy_layers, self.full_support_size, block_output_size_value, block_output_size_policy))

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        min_encoded_state = encoded_state.view(-1, encoded_state.shape[1], encoded_state.shape[2] * encoded_state.shape[3]).min(2, keepdim=True)[0].unsqueeze(-1)
        max_encoded_state = encoded_state.view(-1, encoded_state.shape[1], encoded_state.shape[2] * encoded_state.shape[3]).max(2, keepdim=True)[0].unsqueeze(-1)
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-05] += 1e-05
        encoded_state_normalized = (encoded_state - min_encoded_state) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        action_one_hot = torch.ones((encoded_state.shape[0], 1, encoded_state.shape[2], encoded_state.shape[3])).float()
        action_one_hot = action[:, :, None, None] * action_one_hot / self.action_space_size
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward = self.dynamics_network(x)
        min_next_encoded_state = next_encoded_state.view(-1, next_encoded_state.shape[1], next_encoded_state.shape[2] * next_encoded_state.shape[3]).min(2, keepdim=True)[0].unsqueeze(-1)
        max_next_encoded_state = next_encoded_state.view(-1, next_encoded_state.shape[1], next_encoded_state.shape[2] * next_encoded_state.shape[3]).max(2, keepdim=True)[0].unsqueeze(-1)
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-05] += 1e-05
        next_encoded_state_normalized = (next_encoded_state - min_next_encoded_state) / scale_next_encoded_state
        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        reward = torch.log(torch.zeros(1, self.full_support_size).scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0).repeat(len(observation), 1))
        return value, reward, policy_logits, encoded_state

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DownSample,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualBlock,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_werner_duvaud_muzero_general(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

