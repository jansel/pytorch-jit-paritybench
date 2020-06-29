import sys
_module = sys.modules[__name__]
del sys
conf = _module
ant_v2_ddpg = _module
ant_v2_sac = _module
ant_v2_td3 = _module
continuous_net = _module
discrete_net = _module
halfcheetahBullet_v0_sac = _module
mujoco = _module
maze_env_utils = _module
point = _module
point_maze_env = _module
register = _module
point_maze_td3 = _module
pong_a2c = _module
pong_dqn = _module
pong_ppo = _module
setup = _module
test = _module
base = _module
env = _module
test_batch = _module
test_buffer = _module
test_collector = _module
test_env = _module
continuous = _module
net = _module
test_ddpg = _module
test_ppo = _module
test_sac_with_il = _module
test_td3 = _module
discrete = _module
net = _module
test_a2c_with_il = _module
test_dqn = _module
test_drqn = _module
test_pdqn = _module
test_pg = _module
test_ppo = _module
tianshou = _module
data = _module
batch = _module
buffer = _module
collector = _module
atari = _module
utils = _module
vecenv = _module
exploration = _module
random = _module
policy = _module
base = _module
imitation = _module
base = _module
modelfree = _module
a2c = _module
ddpg = _module
dqn = _module
pg = _module
ppo = _module
sac = _module
td3 = _module
trainer = _module
offpolicy = _module
onpolicy = _module
config = _module
moving_average = _module

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


import torch


import numpy as np


from torch import nn


import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter


from abc import ABC


from abc import abstractmethod


from typing import Dict


from typing import List


from typing import Union


from typing import Optional


from copy import deepcopy


from typing import Tuple


class Actor(nn.Module):

    def __init__(self, layer_num, state_shape, action_shape, max_action,
        device='cpu'):
        super().__init__()
        self.device = device
        self.model = [nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace
            =True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(128, np.prod(action_shape))]
        self.model = nn.Sequential(*self.model)
        self._max = max_action

    def forward(self, s, **kwargs):
        s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        logits = self._max * torch.tanh(logits)
        return logits, None


class ActorProb(nn.Module):

    def __init__(self, layer_num, state_shape, action_shape, max_action,
        device='cpu'):
        super().__init__()
        self.device = device
        self.model = [nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace
            =True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(128, np.prod(action_shape))
        self.sigma = nn.Linear(128, np.prod(action_shape))
        self._max = max_action

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        mu = self._max * torch.tanh(self.mu(logits))
        sigma = torch.exp(self.sigma(logits))
        return (mu, sigma), None


class Critic(nn.Module):

    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [nn.Linear(np.prod(state_shape) + np.prod(action_shape
            ), 128), nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(128, 1)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, a=None):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        if a is not None and not isinstance(a, torch.Tensor):
            a = torch.tensor(a, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        if a is None:
            logits = self.model(s)
        else:
            a = a.view(batch, -1)
            logits = self.model(torch.cat([s, a], dim=1))
        return logits


class Net(nn.Module):

    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace
            =True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        if action_shape:
            self.model += [nn.Linear(128, np.prod(action_shape))]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits, state


class Actor(nn.Module):

    def __init__(self, preprocess_net, action_shape):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(128, np.prod(action_shape))

    def forward(self, s, state=None, info={}):
        logits, h = self.preprocess(s, state)
        logits = F.softmax(self.last(logits), dim=-1)
        return logits, h


class Critic(nn.Module):

    def __init__(self, preprocess_net):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(128, 1)

    def forward(self, s):
        logits, h = self.preprocess(s, None)
        logits = self.last(logits)
        return logits


class DQN(nn.Module):

    def __init__(self, h, w, action_shape, device='cpu'):
        super(DQN, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.fc = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, action_shape)

    def forward(self, x, state=None, info={}):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float)
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.fc(x.reshape(x.size(0), -1))
        return self.head(x), state


class Actor(nn.Module):

    def __init__(self, layer_num, state_shape, action_shape, max_action,
        device='cpu'):
        super().__init__()
        self.device = device
        self.model = [nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace
            =True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(128, np.prod(action_shape))]
        self.model = nn.Sequential(*self.model)
        self._max = max_action

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        logits = self._max * torch.tanh(logits)
        return logits, None


class ActorProb(nn.Module):

    def __init__(self, layer_num, state_shape, action_shape, max_action,
        device='cpu'):
        super().__init__()
        self.device = device
        self.model = [nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace
            =True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*self.model)
        self.mu = nn.Linear(128, np.prod(action_shape))
        self.sigma = nn.Parameter(torch.zeros(np.prod(action_shape), 1))
        self._max = max_action

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        mu = self.mu(logits)
        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), None


class Critic(nn.Module):

    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [nn.Linear(np.prod(state_shape) + np.prod(action_shape
            ), 128), nn.ReLU(inplace=True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        self.model += [nn.Linear(128, 1)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, a=None, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        if a is not None:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, device=self.device, dtype=torch.float)
            a = a.view(batch, -1)
            s = torch.cat([s, a], dim=1)
        logits = self.model(s)
        return logits


class RecurrentActorProb(nn.Module):

    def __init__(self, layer_num, state_shape, action_shape, max_action,
        device='cpu'):
        super().__init__()
        self.device = device
        self.nn = nn.LSTM(input_size=np.prod(state_shape), hidden_size=128,
            num_layers=layer_num, batch_first=True)
        self.mu = nn.Linear(128, np.prod(action_shape))
        self.sigma = nn.Parameter(torch.zeros(np.prod(action_shape), 1))

    def forward(self, s, **kwargs):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        if len(s.shape) == 2:
            bsz, dim = s.shape
            length = 1
        else:
            bsz, length, dim = s.shape
        s = s.view(bsz, length, -1)
        logits, _ = self.nn(s)
        logits = logits[:, (-1)]
        mu = self.mu(logits)
        shape = [1] * len(mu.shape)
        shape[1] = -1
        sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), None


class RecurrentCritic(nn.Module):

    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu'):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.nn = nn.LSTM(input_size=np.prod(state_shape), hidden_size=128,
            num_layers=layer_num, batch_first=True)
        self.fc2 = nn.Linear(128 + np.prod(action_shape), 1)

    def forward(self, s, a=None):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        assert len(s.shape) == 3
        self.nn.flatten_parameters()
        s, (h, c) = self.nn(s)
        s = s[:, (-1)]
        if a is not None:
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, device=self.device, dtype=torch.float)
            s = torch.cat([s, a], dim=1)
        s = self.fc2(s)
        return s


class Net(nn.Module):

    def __init__(self, layer_num, state_shape, action_shape=0, device='cpu',
        softmax=False):
        super().__init__()
        self.device = device
        self.model = [nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace
            =True)]
        for i in range(layer_num):
            self.model += [nn.Linear(128, 128), nn.ReLU(inplace=True)]
        if action_shape:
            self.model += [nn.Linear(128, np.prod(action_shape))]
        if softmax:
            self.model += [nn.Softmax(dim=-1)]
        self.model = nn.Sequential(*self.model)

    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits, state


class Actor(nn.Module):

    def __init__(self, preprocess_net, action_shape):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(128, np.prod(action_shape))

    def forward(self, s, state=None, info={}):
        logits, h = self.preprocess(s, state)
        logits = F.softmax(self.last(logits), dim=-1)
        return logits, h


class Critic(nn.Module):

    def __init__(self, preprocess_net):
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(128, 1)

    def forward(self, s, **kwargs):
        logits, h = self.preprocess(s, state=kwargs.get('state', None))
        logits = self.last(logits)
        return logits


class Recurrent(nn.Module):

    def __init__(self, layer_num, state_shape, action_shape, device='cpu'):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.fc1 = nn.Linear(np.prod(state_shape), 128)
        self.nn = nn.LSTM(input_size=128, hidden_size=128, num_layers=
            layer_num, batch_first=True)
        self.fc2 = nn.Linear(128, np.prod(action_shape))

    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        if len(s.shape) == 2:
            bsz, dim = s.shape
            length = 1
        else:
            bsz, length, dim = s.shape
        s = self.fc1(s.view([bsz * length, dim]))
        s = s.view(bsz, length, -1)
        self.nn.flatten_parameters()
        if state is None:
            s, (h, c) = self.nn(s)
        else:
            s, (h, c) = self.nn(s, (state['h'].transpose(0, 1).contiguous(),
                state['c'].transpose(0, 1).contiguous()))
        s = self.fc2(s[:, (-1)])
        return s, {'h': h.transpose(0, 1).detach(), 'c': c.transpose(0, 1).
            detach()}


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_thu_ml_tianshou(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ActorProb(*[], **{'layer_num': 1, 'state_shape': 4, 'action_shape': 4, 'max_action': 4}), [torch.rand([4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Net(*[], **{'layer_num': 1, 'state_shape': 4}), [torch.rand([4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(Recurrent(*[], **{'layer_num': 1, 'state_shape': 4, 'action_shape': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(RecurrentActorProb(*[], **{'layer_num': 1, 'state_shape': 4, 'action_shape': 4, 'max_action': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(RecurrentCritic(*[], **{'layer_num': 1, 'state_shape': 4}), [torch.rand([4, 4, 4])], {})

