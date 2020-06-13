import sys
_module = sys.modules[__name__]
del sys
Base_Agent = _module
DDQN = _module
DDQN_With_Prioritised_Experience_Replay = _module
DQN = _module
DQN_HER = _module
DQN_With_Fixed_Q_Targets = _module
Dueling_DDQN = _module
HER_Base = _module
Trainer = _module
A2C = _module
A3C = _module
DDPG = _module
DDPG_HER = _module
SAC = _module
SAC_Discrete = _module
TD3 = _module
DIAYN = _module
HIRO = _module
SNN_HRL = _module
h_DQN = _module
PPO = _module
REINFORCE = _module
Ant_Navigation_Environments = _module
Atari_Environment = _module
Bit_Flipping_Environment = _module
Four_Rooms_Environment = _module
Long_Corridor_Environment = _module
Open_AI_Wrappers = _module
ant_environments = _module
ant = _module
ant_maze_env = _module
create_maze_env = _module
maze_env = _module
maze_env_utils = _module
point = _module
point_maze_env = _module
Base_Exploration_Strategy = _module
Epsilon_Greedy_Exploration = _module
Gaussian_Exploration = _module
OU_Noise_Exploration = _module
Bit_Flipping = _module
Cart_Pole = _module
Fetch_Reach = _module
Four_Rooms = _module
HRL_Experiments = _module
Hopper = _module
Long_Corridor = _module
Mountain_Car = _module
Reacher = _module
Space_Invaders = _module
Taxi = _module
Walker = _module
Plot_Sets_Of_Results = _module
Test_Action_Balanced_Replay_Buffer = _module
Test_Agents = _module
Test_Bit_Flipping_Environment = _module
Test_DQN_HER = _module
Test_Deque = _module
Test_Four_Rooms_Environment = _module
Test_HIRO = _module
Test_HRL = _module
Test_Max_Heap = _module
Test_Memory_Shaper = _module
Test_Prioritised_Replay_Buffer = _module
Test_Sequitur = _module
Test_Trainer = _module
Deepmind_RMS_Prop = _module
Memory_Shaper = _module
OU_Noise = _module
Parallel_Experience_Generator = _module
Tensorboard = _module
Utility_Functions = _module
Action_Balanced_Replay_Buffer = _module
Config = _module
Deque = _module
Max_Heap = _module
Node = _module
Prioritised_Replay_Buffer = _module
Replay_Buffer = _module
Tanh_Distribution = _module
k_Sequitur = _module

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


import logging


import random


import numpy as np


import torch


from torch.optim import optimizer


import torch.nn.functional as F


from collections import Counter


import torch.optim as optim


import copy


from torch import multiprocessing


from torch.multiprocessing import Queue


from torch.optim import Adam


import torch.nn.functional as functional


from torch import optim


from torch.distributions import Normal


from torch import nn


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_p_christ_Deep_Reinforcement_Learning_Algorithms_with_PyTorch(_paritybench_base):
    pass
