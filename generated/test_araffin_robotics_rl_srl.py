import sys
_module = sys.modules[__name__]
del sys
conftest = _module
conf = _module
environments = _module
car_racing = _module
car_env = _module
change_to_relative_pos = _module
dataset_fusioner = _module
dataset_generator = _module
button_world = _module
kuka_env = _module
gym_baxter = _module
baxter_env = _module
test_baxter_env = _module
kuka_gym = _module
kuka = _module
kuka_2button_gym_env = _module
kuka_button_gym_env = _module
kuka_moving_button_gym_env = _module
kuka_rand_button_gym_env = _module
mobile_robot = _module
mobile_robot_1D_env = _module
mobile_robot_2target_env = _module
mobile_robot_env = _module
mobile_robot_line_target_env = _module
test_env = _module
omnirobot_gym = _module
omnirobot_env = _module
registry = _module
robobo_gym = _module
robobo_env = _module
srl_env = _module
utils = _module
real_robots = _module
constants = _module
gazebo_server = _module
omnirobot_server = _module
omnirobot_simulator_server = _module
omnirobot_utils = _module
marker_finder = _module
marker_render = _module
omnirobot_manager_base = _module
real_baxter_debug = _module
real_baxter_server = _module
real_robobo_server = _module
teleop_client = _module
replay = _module
aggregate_plots = _module
compare_plots = _module
enjoy_baselines = _module
gather_results = _module
plots = _module
rl_baselines = _module
base_classes = _module
evolution_strategies = _module
ars = _module
cma_es = _module
hyperparam_search = _module
pipeline = _module
random_agent = _module
rl_algorithm = _module
a2c = _module
acer = _module
acktr = _module
ddpg = _module
deepq = _module
ppo1 = _module
ppo2 = _module
sac = _module
trpo = _module
train = _module
utils = _module
visualize = _module
state_representation = _module
client = _module
episode_saver = _module
models = _module
tests = _module
test_dataset_manipulation = _module
test_end_to_end = _module
test_enjoy = _module
test_hyperparam_search = _module
test_pipeline = _module

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


import numpy as np


import torch as th


import matplotlib.pyplot as plt


import time


from scipy.spatial.transform import Rotation as R


import torch


import torch.nn as nn


import torch.nn.functional as F


from collections import OrderedDict


import tensorflow as tf


class CNNPolicyPytorch(nn.Module):
    """
    A simple CNN policy using pytorch
    :param out_dim: (int)
    """

    def __init__(self, in_dim, out_dim):
        super(CNNPolicyPytorch, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 8, kernel_size=5, padding=2, stride=2, bias=False)
        self.norm1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2, bias=False)
        self.norm2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False)
        self.norm3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2)
        self.fc = nn.Linear(288, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MLPPolicyPytorch(nn.Module):
    """
    A simple MLP policy using pytorch
    :param in_dim: (int)
    :param hidden_dims: ([int])
    :param out_dim: (int)
    """

    def __init__(self, in_dim, hidden_dims, out_dim):
        super(MLPPolicyPytorch, self).__init__()
        self.fc_hidden_name = []
        self.fc_in = nn.Linear(int(in_dim), int(hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.add_module('fc_' + str(i), nn.Linear(int(hidden_dims[i]), int(hidden_dims[i + 1])))
            self.fc_hidden_name.append('fc_' + str(i))
        self.fc_out = nn.Linear(int(hidden_dims[-1]), int(out_dim))

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        for name in self.fc_hidden_name:
            x = F.relu(getattr(self, name)(x))
        x = self.fc_out(x)
        return x

