import sys
_module = sys.modules[__name__]
del sys
conf = _module
dm_manipulation_stack_sac = _module
dm_suite_cartpole_swingup_ddpg = _module
gym_cartpole_cem = _module
gym_cartpole_cem_eval = _module
gym_cartpole_dqn = _module
gym_cartpole_dqn_eval = _module
gym_frozen_lake_q_learning = _module
gym_frozen_lake_q_learning_eval = _module
gym_pendulum_ddpg = _module
gym_pendulum_ddpg_eval = _module
gym_taxi_sarsa = _module
gym_taxi_sarsa_eval = _module
gym_vector_cartpole_dqn = _module
gym_vector_frozen_lake_q_learning = _module
gym_vector_pendulum_ddpg = _module
gym_vector_taxi_sarsa = _module
amp_humanoid = _module
isaacgym_parallel_no_shared_memory = _module
isaacgym_parallel_no_shared_memory_eval = _module
isaacgym_sequential_no_shared_memory = _module
isaacgym_sequential_no_shared_memory_eval = _module
isaacgym_sequential_shared_memory = _module
isaacgym_sequential_shared_memory_eval = _module
ppo_allegro_hand = _module
ppo_ant = _module
ppo_anymal = _module
ppo_anymal_terrain = _module
ppo_ball_balance = _module
ppo_cartpole = _module
ppo_cartpole_eval = _module
ppo_franka_cabinet = _module
ppo_humanoid = _module
ppo_ingenuity = _module
ppo_quadcopter = _module
ppo_shadow_hand = _module
ppo_trifinger = _module
trpo_cartpole = _module
cartpole_example_skrl = _module
isaacsim_jetbot_ppo = _module
ppo_allegro_hand = _module
ppo_ant = _module
ppo_ant_mt = _module
ppo_anymal = _module
ppo_anymal_terrain = _module
ppo_ball_balance = _module
ppo_cartpole = _module
ppo_cartpole_mt = _module
ppo_crazy_flie = _module
ppo_franka_cabinet = _module
ppo_humanoid = _module
ppo_ingenuity = _module
ppo_quadcopter = _module
ppo_shadow_hand = _module
reaching_franka_isaacgym_env = _module
reaching_franka_isaacgym_skrl_eval = _module
reaching_franka_isaacgym_skrl_train = _module
reaching_franka_omniverse_isaacgym_env = _module
reaching_franka_omniverse_isaacgym_skrl_eval = _module
reaching_franka_omniverse_isaacgym_skrl_train = _module
reaching_franka_real_env = _module
reaching_franka_real_skrl_eval = _module
tensorboard_file_iterator = _module
agent = _module
categorical_model = _module
deterministic_model = _module
gaussian_model = _module
isaacgym_utils = _module
memory = _module
model_mixin = _module
multivariate_gaussian_model = _module
noise = _module
shared_model = _module
tabular_model = _module
trainer = _module
utils_postprocessing = _module
setup = _module
skrl = _module
agents = _module
a2c = _module
a2c = _module
amp = _module
amp = _module
base = _module
cem = _module
cem = _module
ddpg = _module
ddpg = _module
dqn = _module
ddqn = _module
dqn = _module
ppo = _module
ppo = _module
q_learning = _module
q_learning = _module
sac = _module
sac = _module
sarsa = _module
sarsa = _module
td3 = _module
td3 = _module
trpo = _module
trpo = _module
envs = _module
loaders = _module
wrappers = _module
memories = _module
base = _module
prioritized = _module
random = _module
models = _module
base = _module
categorical = _module
deterministic = _module
gaussian = _module
multivariate_gaussian = _module
tabular = _module
resources = _module
noises = _module
base = _module
gaussian = _module
ornstein_uhlenbeck = _module
preprocessors = _module
running_standard_scaler = _module
schedulers = _module
kl_adaptive = _module
trainers = _module
base = _module
manual = _module
parallel = _module
sequential = _module
utils = _module
control = _module
isaacgym_utils = _module
model_instantiators = _module
omniverse_isaacgym_utils = _module
postprocessing = _module
tests = _module
test_noises_gaussian = _module
test_noises_ornstein_uhlenbeck = _module

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


import numpy as np


from typing import Union


from typing import Tuple


from typing import Dict


from typing import Any


from typing import List


from typing import Optional


from typing import Sequence


import copy


import itertools


from typing import Callable


import math


from typing import Mapping


import collections


from torch.utils.tensorboard import SummaryWriter


from torch.nn.utils.convert_parameters import parameters_to_vector


from torch.nn.utils.convert_parameters import vector_to_parameters


import queue


import functools


from torch.utils.data.sampler import BatchSampler


from torch.distributions import Categorical


from torch.distributions import Normal


from torch.distributions import MultivariateNormal


from torch.optim.lr_scheduler import _LRScheduler


import torch.multiprocessing as mp


import time


import random


import logging


from enum import Enum


logger = logging.getLogger('skrl')

