import sys
_module = sys.modules[__name__]
del sys
MADDPG = _module
main = _module
memory = _module
model = _module
params = _module
pursuit = _module
policy_eval = _module
speed_pursuit = _module
stationary_eval = _module
pursuit_evade = _module
AgentLayer = _module
Controllers = _module
DiscreteAgent = _module
TwoDMaps = _module
utils = _module
agent_utils = _module
vis_policy = _module
waterworld = _module
waterworld_modified = _module
randomProcess = _module

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


import torch as th


from copy import deepcopy


from torch.optim import Adam


import torch.nn as nn


import numpy as np


import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = self.dim_action * n_agent
        self.FC1 = nn.Linear(obs_dim, 1024)
        self.FC2 = nn.Linear(1024 + act_dim, 512)
        self.FC3 = nn.Linear(512, 300)
        self.FC4 = nn.Linear(300, 1)

    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = th.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))


class Actor(nn.Module):

    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 500)
        self.FC2 = nn.Linear(500, 128)
        self.FC3 = nn.Linear(128, dim_action)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.tanh(self.FC3(result))
        return result


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_xuehy_pytorch_maddpg(_paritybench_base):
    pass
    def test_000(self):
        self._check(Actor(*[], **{'dim_observation': 4, 'dim_action': 4}), [torch.rand([4, 4, 4, 4])], {})

