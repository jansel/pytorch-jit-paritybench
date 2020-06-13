import sys
_module = sys.modules[__name__]
del sys
abstract_game = _module
breakout = _module
cartpole = _module
connect4 = _module
gomoku = _module
gridworld = _module
lunarlander = _module
tictactoe = _module
twentyone = _module
models = _module
muzero = _module
replay_buffer = _module
self_play = _module
shared_storage = _module
trainer = _module

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


import math


from abc import ABC


from abc import abstractmethod


import torch


import numpy


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
        return {key: value.cpu() for key, value in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride
        =stride, padding=1, bias=False)


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
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels // 2,
            kernel_size=3, stride=2, padding=1, bias=False)
        self.resblocks1 = torch.nn.ModuleList([ResidualBlock(out_channels //
            2) for _ in range(2)])
        self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels,
            kernel_size=3, stride=2, padding=1, bias=False)
        self.resblocks2 = torch.nn.ModuleList([ResidualBlock(out_channels) for
            _ in range(3)])
        self.pooling1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = torch.nn.ModuleList([ResidualBlock(out_channels) for
            _ in range(3)])
        self.pooling2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        for block in self.resblocks1:
            out = block(out)
        out = self.conv2(out)
        for block in self.resblocks2:
            out = block(out)
        out = self.pooling1(out)
        for block in self.resblocks3:
            out = block(out)
        out = self.pooling2(out)
        return out


class RepresentationNetwork(torch.nn.Module):

    def __init__(self, observation_shape, stacked_observations, num_blocks,
        num_channels, downsample):
        super().__init__()
        self.use_downsample = downsample
        if self.use_downsample:
            self.downsample = DownSample(observation_shape[0] * (
                stacked_observations + 1) + stacked_observations, num_channels)
        self.conv = conv3x3(observation_shape[0] * (stacked_observations + 
            1) + stacked_observations, num_channels)
        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.resblocks = torch.nn.ModuleList([ResidualBlock(num_channels) for
            _ in range(num_blocks)])

    def forward(self, x):
        if self.use_downsample:
            out = self.downsample(x)
        else:
            out = self.conv(x)
            out = self.bn(out)
            out = torch.nn.functional.relu(out)
        for block in self.resblocks:
            out = block(out)
        return out


class DynamicsNetwork(torch.nn.Module):

    def __init__(self, num_blocks, num_channels, reduced_channels_reward,
        fc_reward_layers, full_support_size, block_output_size_reward):
        super().__init__()
        self.conv = conv3x3(num_channels, num_channels - 1)
        self.bn = torch.nn.BatchNorm2d(num_channels - 1)
        self.resblocks = torch.nn.ModuleList([ResidualBlock(num_channels - 
            1) for _ in range(num_blocks)])
        self.conv1x1_reward = torch.nn.Conv2d(num_channels - 1,
            reduced_channels_reward, 1)
        self.block_output_size_reward = block_output_size_reward
        self.fc = FullyConnectedNetwork(self.block_output_size_reward,
            fc_reward_layers, full_support_size)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = torch.nn.functional.relu(out)
        for block in self.resblocks:
            out = block(out)
        state = out
        out = self.conv1x1_reward(out)
        out = out.view(-1, self.block_output_size_reward)
        reward = self.fc(out)
        return state, reward


class PredictionNetwork(torch.nn.Module):

    def __init__(self, action_space_size, num_blocks, num_channels,
        reduced_channels_value, reduced_channels_policy, fc_value_layers,
        fc_policy_layers, full_support_size, block_output_size_value,
        block_output_size_policy):
        super().__init__()
        self.resblocks = torch.nn.ModuleList([ResidualBlock(num_channels) for
            _ in range(num_blocks)])
        self.conv1x1_value = torch.nn.Conv2d(num_channels,
            reduced_channels_value, 1)
        self.conv1x1_policy = torch.nn.Conv2d(num_channels,
            reduced_channels_policy, 1)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = FullyConnectedNetwork(self.block_output_size_value,
            fc_value_layers, full_support_size)
        self.fc_policy = FullyConnectedNetwork(self.
            block_output_size_policy, fc_policy_layers, action_space_size)

    def forward(self, x):
        out = x
        for block in self.resblocks:
            out = block(out)
        value = self.conv1x1_value(out)
        policy = self.conv1x1_policy(out)
        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


class FullyConnectedNetwork(torch.nn.Module):

    def __init__(self, input_size, layer_sizes, output_size, activation=None):
        super().__init__()
        size_list = [input_size] + layer_sizes
        layers = []
        if 1 < len(size_list):
            for i in range(len(size_list) - 1):
                layers.extend([torch.nn.Linear(size_list[i], size_list[i + 
                    1]), torch.nn.LeakyReLU()])
        layers.append(torch.nn.Linear(size_list[-1], output_size))
        if activation:
            layers.append(activation)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_werner_duvaud_muzero_general(_paritybench_base):
    pass
    def test_000(self):
        self._check(DownSample(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(RepresentationNetwork(*[], **{'observation_shape': [4, 4], 'stacked_observations': 4, 'num_blocks': 1, 'num_channels': 4, 'downsample': 4}), [torch.rand([4, 24, 64, 64])], {})

    def test_002(self):
        self._check(ResidualBlock(*[], **{'num_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

