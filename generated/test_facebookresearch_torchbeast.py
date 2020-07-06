import sys
_module = sys.modules[__name__]
del sys
nest_test = _module
setup = _module
setup = _module
batching_queue_test = _module
contiguous_arrays_env = _module
contiguous_arrays_test = _module
core_agent_state_env = _module
core_agent_state_test = _module
dynamic_batcher_test = _module
inference_speed_profiling = _module
polybeast_inference_test = _module
polybeast_learn_function_test = _module
polybeast_loss_functions_test = _module
polybeast_net_test = _module
vtrace_test = _module
atari_wrappers = _module
environment = _module
file_writer = _module
prof = _module
vtrace = _module
monobeast = _module
polybeast = _module
polybeast_env = _module

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


from torch.utils import cpp_extension


import time


from torch import nn


import logging


import warnings


import copy


from torch.nn import functional as F


import collections


import torch.nn.functional as F


import typing


from torch import multiprocessing as mp


class Net(nn.Module):

    def __init__(self, num_actions, use_lstm=False):
        super(Net, self).__init__()
        self.num_actions = num_actions
        self.use_lstm = use_lstm
        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []
        input_channels = 4
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))
            input_channels = num_ch
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                resnet_block.append(nn.ReLU())
                resnet_block.append(nn.Conv2d(in_channels=input_channels, out_channels=num_ch, kernel_size=3, stride=1, padding=1))
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))
        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)
        self.fc = nn.Linear(3872, 256)
        core_output_size = self.fc.out_features + 1
        if use_lstm:
            self.core = nn.LSTM(core_output_size, 256, num_layers=1)
            core_output_size = 256
        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size) for _ in range(2))

    def forward(self, inputs, core_state):
        x = inputs['frame']
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)
        x = x.float() / 255.0
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input
        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))
        clipped_reward = torch.clamp(inputs['reward'], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)
        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs['done']).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                nd = nd.view(1, -1, 1)
                core_state = nest.map(nd.mul, core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)
        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        return (action, policy_logits, baseline), core_state


class AtariNet(nn.Module):

    def __init__(self, observation_shape, num_actions, use_lstm=False):
        super(AtariNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(in_channels=self.observation_shape[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        core_output_size = self.fc.out_features + num_actions + 1
        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, 2)
        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size) for _ in range(2))

    def forward(self, inputs, core_state=()):
        x = inputs['frame']
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))
        one_hot_last_action = F.one_hot(inputs['last_action'].view(T * B), self.num_actions).float()
        clipped_reward = torch.clamp(inputs['reward'], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)
        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs['done']).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)
        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        return dict(policy_logits=policy_logits, baseline=baseline, action=action), core_state

