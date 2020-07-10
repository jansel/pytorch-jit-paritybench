import sys
_module = sys.modules[__name__]
del sys
plot_benchmark = _module
plot_script = _module
run_lab = _module
setup = _module
slm_lab = _module
agent = _module
algorithm = _module
actor_critic = _module
base = _module
dqn = _module
policy_util = _module
ppo = _module
random = _module
reinforce = _module
sac = _module
sarsa = _module
sil = _module
memory = _module
onpolicy = _module
prioritized = _module
replay = _module
net = _module
base = _module
conv = _module
mlp = _module
net_util = _module
q_net = _module
recurrent = _module
env = _module
openai = _module
registration = _module
unity = _module
vec_env = _module
vizdoom = _module
cfgs = _module
vizdoom_env = _module
wrapper = _module
experiment = _module
analysis = _module
control = _module
retro_analysis = _module
search = _module
lib = _module
decorator = _module
distribution = _module
logger = _module
math_util = _module
optimizer = _module
util = _module
viz = _module
spec = _module
random_baseline = _module
spec_util = _module
test = _module
test_onpolicy_memory = _module
test_per_memory = _module
test_replay_memory = _module
test_conv = _module
test_mlp = _module
test_recurrent = _module
conftest = _module
test_registration = _module
test_vec_env = _module
test_wrapper = _module
test_control = _module
test_monitor = _module
test_distribution = _module
test_logger = _module
test_math_util = _module
test_util = _module
test_dist_spec = _module
test_spec = _module
test_spec_util = _module

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
xrange = range
wraps = functools.wraps


import torch.multiprocessing as mp


from torch.utils.tensorboard import SummaryWriter


import numpy as np


import torch


import warnings


from torch import distributions


from copy import deepcopy


import math


import torch.nn.functional as F


from abc import ABC


from abc import abstractmethod


import torch.nn as nn


from functools import partial


from functools import wraps


from collections import OrderedDict


import random


from torch.optim.optimizer import Optimizer


import itertools as it


from collections import deque


import time

