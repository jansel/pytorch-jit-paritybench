import sys
_module = sys.modules[__name__]
del sys
core = _module
agent = _module
agent_single_process = _module
agents = _module
a3c = _module
a3c_single_process = _module
acer = _module
acer_single_process = _module
dqn = _module
empty = _module
env = _module
envs = _module
atari = _module
atari_ram = _module
gym = _module
lab = _module
memories = _module
episode_parameter = _module
episodic = _module
sequential = _module
memory = _module
model = _module
models = _module
a3c_cnn_dis = _module
a3c_mlp_con = _module
acer_cnn_dis = _module
acer_mlp_dis = _module
dqn_cnn = _module
dqn_mlp = _module
empty = _module
main = _module
optims = _module
helpers = _module
sharedAdam = _module
sharedRMSprop = _module
utils = _module
distributions = _module
factory = _module
init_weights = _module
options = _module

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


import torch.optim as optim


import numpy as np


import random


import time


import math


from torch.autograd import Variable


import torch.nn.functional as F


import torch.multiprocessing as mp


from torch.autograd import grad


from torch.autograd import backward


import torch.nn as nn


from torch import optim


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.logger = args.logger
        self.hidden_dim = args.hidden_dim
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        if hasattr(args, 'enable_dueling'):
            self.enable_dueling = args.enable_dueling
            self.dueling_type = args.dueling_type
        if hasattr(args, 'enable_lstm'):
            self.enable_lstm = args.enable_lstm
        self.input_dims = {}
        self.input_dims[0] = args.hist_len
        self.input_dims[1] = args.state_shape
        self.output_dims = args.action_dim

    def _init_weights(self):
        raise NotImplementedError('not implemented in base calss')

    def print_model(self):
        self.logger.warning('<-----------------------------------> Model')
        self.logger.warning(self)

    def _reset(self):
        self._init_weights()
        self.type(self.dtype)
        self.print_model()

    def forward(self, input):
        raise NotImplementedError('not implemented in base calss')


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
    return out


class A3CCnnDisModel(Model):

    def __init__(self, args):
        super(A3CCnnDisModel, self).__init__(args)
        self.conv1 = nn.Conv2d(self.input_dims[0], 32, kernel_size=3, stride=2)
        self.rl1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.rl2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.rl3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.rl4 = nn.ReLU()
        if self.enable_lstm:
            self.lstm = nn.LSTMCell(3 * 3 * 32, self.hidden_dim)
        self.policy_5 = nn.Linear(self.hidden_dim, self.output_dims)
        self.policy_6 = nn.Softmax()
        self.value_5 = nn.Linear(self.hidden_dim, 1)
        self._reset()

    def _init_weights(self):
        self.apply(init_weights)
        self.policy_5.weight.data = normalized_columns_initializer(self.policy_5.weight.data, 0.01)
        self.policy_5.bias.data.fill_(0)
        self.value_5.weight.data = normalized_columns_initializer(self.value_5.weight.data, 1.0)
        self.value_5.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, x, lstm_hidden_vb=None):
        x = x.view(x.size(0), self.input_dims[0], self.input_dims[1], self.input_dims[1])
        x = self.rl1(self.conv1(x))
        x = self.rl2(self.conv2(x))
        x = self.rl3(self.conv3(x))
        x = self.rl4(self.conv4(x))
        x = x.view(-1, 3 * 3 * 32)
        if self.enable_lstm:
            x, c = self.lstm(x, lstm_hidden_vb)
        p = self.policy_5(x)
        p = self.policy_6(p)
        v = self.value_5(x)
        if self.enable_lstm:
            return p, v, (x, c)
        else:
            return p, v


class A3CMlpConModel(Model):

    def __init__(self, args):
        super(A3CMlpConModel, self).__init__(args)
        self.fc1 = nn.Linear(self.input_dims[0] * self.input_dims[1], self.hidden_dim)
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl3 = nn.ReLU()
        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl4 = nn.ReLU()
        self.fc1_v = nn.Linear(self.input_dims[0] * self.input_dims[1], self.hidden_dim)
        self.rl1_v = nn.ReLU()
        self.fc2_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl2_v = nn.ReLU()
        self.fc3_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl3_v = nn.ReLU()
        self.fc4_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl4_v = nn.ReLU()
        if self.enable_lstm:
            self.lstm = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
            self.lstm_v = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.policy_5 = nn.Linear(self.hidden_dim, self.output_dims)
        self.policy_sig = nn.Linear(self.hidden_dim, self.output_dims)
        self.softplus = nn.Softplus()
        self.value_5 = nn.Linear(self.hidden_dim, 1)
        self._reset()

    def _init_weights(self):
        self.apply(init_weights)
        self.fc1.weight.data = normalized_columns_initializer(self.fc1.weight.data, 0.01)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data = normalized_columns_initializer(self.fc2.weight.data, 0.01)
        self.fc2.bias.data.fill_(0)
        self.fc3.weight.data = normalized_columns_initializer(self.fc3.weight.data, 0.01)
        self.fc3.bias.data.fill_(0)
        self.fc4.weight.data = normalized_columns_initializer(self.fc4.weight.data, 0.01)
        self.fc4.bias.data.fill_(0)
        self.policy_5.weight.data = normalized_columns_initializer(self.policy_5.weight.data, 0.01)
        self.policy_5.bias.data.fill_(0)
        self.value_5.weight.data = normalized_columns_initializer(self.value_5.weight.data, 1.0)
        self.value_5.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.lstm_v.bias_ih.data.fill_(0)
        self.lstm_v.bias_hh.data.fill_(0)

    def forward(self, x, lstm_hidden_vb=None):
        p = x.view(x.size(0), self.input_dims[0] * self.input_dims[1])
        p = self.rl1(self.fc1(p))
        p = self.rl2(self.fc2(p))
        p = self.rl3(self.fc3(p))
        p = self.rl4(self.fc4(p))
        p = p.view(-1, self.hidden_dim)
        if self.enable_lstm:
            p_, v_ = torch.split(lstm_hidden_vb[0], 1)
            c_p, c_v = torch.split(lstm_hidden_vb[1], 1)
            p, c_p = self.lstm(p, (p_, c_p))
        p_out = self.policy_5(p)
        sig = self.policy_sig(p)
        sig = self.softplus(sig)
        v = x.view(x.size(0), self.input_dims[0] * self.input_dims[1])
        v = self.rl1_v(self.fc1_v(v))
        v = self.rl2_v(self.fc2_v(v))
        v = self.rl3_v(self.fc3_v(v))
        v = self.rl4_v(self.fc4_v(v))
        v = v.view(-1, self.hidden_dim)
        if self.enable_lstm:
            v, c_v = self.lstm_v(v, (v_, c_v))
        v_out = self.value_5(v)
        if self.enable_lstm:
            return p_out, sig, v_out, (torch.cat((p, v), 0), torch.cat((c_p, c_v), 0))
        else:
            return p_out, sig, v_out


class ACERCnnDisModel(Model):

    def __init__(self, args):
        super(ACERCnnDisModel, self).__init__(args)
        self.conv1 = nn.Conv2d(self.input_dims[0], 32, kernel_size=3, stride=2)
        self.rl1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.rl2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.rl3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.rl4 = nn.ReLU()
        if self.enable_lstm:
            self.lstm = nn.LSTMCell(3 * 3 * 32, self.hidden_dim)
        self.actor_5 = nn.Linear(self.hidden_dim, self.output_dims)
        self.actor_6 = nn.Softmax()
        self.critic_5 = nn.Linear(self.hidden_dim, self.output_dims)
        self._reset()

    def _init_weights(self):
        self.apply(init_weights)
        self.actor_5.weight.data = normalized_columns_initializer(self.actor_5.weight.data, 0.01)
        self.actor_5.bias.data.fill_(0)
        self.critic_5.weight.data = normalized_columns_initializer(self.critic_5.weight.data, 1.0)
        self.critic_5.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, x, lstm_hidden_vb=None):
        x = x.view(x.size(0), self.input_dims[0], self.input_dims[1], self.input_dims[1])
        x = self.rl1(self.conv1(x))
        x = self.rl2(self.conv2(x))
        x = self.rl3(self.conv3(x))
        x = self.rl4(self.conv4(x))
        x = x.view(-1, 3 * 3 * 32)
        if self.enable_lstm:
            x, c = self.lstm(x, lstm_hidden_vb)
        policy = self.actor_6(self.actor_5(x)).clamp(max=1 - 1e-06, min=1e-06)
        q = self.critic_5(x)
        v = (q * policy).sum(1, keepdim=True)
        if self.enable_lstm:
            return policy, q, v, (x, c)
        else:
            return policy, q, v


class ACERMlpDisModel(Model):

    def __init__(self, args):
        super(ACERMlpDisModel, self).__init__(args)
        self.fc1 = nn.Linear(self.input_dims[0] * self.input_dims[1], self.hidden_dim)
        self.rl1 = nn.ReLU()
        if self.enable_lstm:
            self.lstm = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.actor_2 = nn.Linear(self.hidden_dim, self.output_dims)
        self.actor_3 = nn.Softmax()
        self.critic_2 = nn.Linear(self.hidden_dim, self.output_dims)
        self._reset()

    def _init_weights(self):
        self.apply(init_weights)
        self.actor_2.weight.data = normalized_columns_initializer(self.actor_2.weight.data, 0.01)
        self.actor_2.bias.data.fill_(0)
        self.critic_2.weight.data = normalized_columns_initializer(self.critic_2.weight.data, 1.0)
        self.critic_2.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, x, lstm_hidden_vb=None):
        x = x.view(x.size(0), self.input_dims[0] * self.input_dims[1])
        x = self.rl1(self.fc1(x))
        if self.enable_lstm:
            x, c = self.lstm(x, lstm_hidden_vb)
        policy = self.actor_3(self.actor_2(x)).clamp(max=1 - 1e-06, min=1e-06)
        q = self.critic_2(x)
        v = (q * policy).sum(1, keepdim=True)
        if self.enable_lstm:
            return policy, q, v, (x, c)
        else:
            return policy, q, v


class DQNCnnModel(Model):

    def __init__(self, args):
        super(DQNCnnModel, self).__init__(args)
        self.conv1 = nn.Conv2d(self.input_dims[0], 32, kernel_size=3, stride=2)
        self.rl1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.rl2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.rl3 = nn.ReLU()
        self.fc4 = nn.Linear(32 * 5 * 5, self.hidden_dim)
        self.rl4 = nn.ReLU()
        if self.enable_dueling:
            self.fc5 = nn.Linear(self.hidden_dim, self.output_dims + 1)
            self.v_ind = torch.LongTensor(self.output_dims).fill_(0).unsqueeze(0)
            self.a_ind = torch.LongTensor(np.arange(1, self.output_dims + 1)).unsqueeze(0)
        else:
            self.fc5 = nn.Linear(self.hidden_dim, self.output_dims)
        self._reset()

    def _init_weights(self):
        self.apply(init_weights)
        self.fc4.weight.data = normalized_columns_initializer(self.fc4.weight.data, 0.0001)
        self.fc4.bias.data.fill_(0)
        self.fc5.weight.data = normalized_columns_initializer(self.fc5.weight.data, 0.0001)
        self.fc5.bias.data.fill_(0)

    def forward(self, x):
        x = x.view(x.size(0), self.input_dims[0], self.input_dims[1], self.input_dims[1])
        x = self.rl1(self.conv1(x))
        x = self.rl2(self.conv2(x))
        x = self.rl3(self.conv3(x))
        x = self.rl4(self.fc4(x.view(x.size(0), -1)))
        if self.enable_dueling:
            x = self.fc5(x)
            v_ind_vb = Variable(self.v_ind)
            a_ind_vb = Variable(self.a_ind)
            if self.use_cuda:
                v_ind_vb = v_ind_vb
                a_ind_vb = a_ind_vb
            v = x.gather(1, v_ind_vb.expand(x.size(0), self.output_dims))
            a = x.gather(1, a_ind_vb.expand(x.size(0), self.output_dims))
            if self.dueling_type == 'avg':
                x = v + (a - a.mean(1).expand(x.size(0), self.output_dims))
            elif self.dueling_type == 'max':
                x = v + (a - a.max(1)[0].expand(x.size(0), self.output_dims))
            elif self.dueling_type == 'naive':
                x = v + a
            else:
                assert False, "dueling_type must be one of {'avg', 'max', 'naive'}"
            del v_ind_vb, a_ind_vb, v, a
            return x
        else:
            return self.fc5(x.view(x.size(0), -1))


class DQNMlpModel(Model):

    def __init__(self, args):
        super(DQNMlpModel, self).__init__(args)
        self.fc1 = nn.Linear(self.input_dims[0] * self.input_dims[1], self.hidden_dim)
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.rl3 = nn.ReLU()
        if self.enable_dueling:
            self.fc4 = nn.Linear(self.hidden_dim, self.output_dims + 1)
            self.v_ind = torch.LongTensor(self.output_dims).fill_(0).unsqueeze(0)
            self.a_ind = torch.LongTensor(np.arange(1, self.output_dims + 1)).unsqueeze(0)
        else:
            self.fc4 = nn.Linear(self.hidden_dim, self.output_dims)
        self._reset()

    def _init_weights(self):
        pass

    def forward(self, x):
        x = x.view(x.size(0), self.input_dims[0] * self.input_dims[1])
        x = self.rl1(self.fc1(x))
        x = self.rl2(self.fc2(x))
        x = self.rl3(self.fc3(x))
        if self.enable_dueling:
            x = self.fc4(x.view(x.size(0), -1))
            v_ind_vb = Variable(self.v_ind)
            a_ind_vb = Variable(self.a_ind)
            if self.use_cuda:
                v_ind_vb = v_ind_vb
                a_ind_vb = a_ind_vb
            v = x.gather(1, v_ind_vb.expand(x.size(0), self.output_dims))
            a = x.gather(1, a_ind_vb.expand(x.size(0), self.output_dims))
            if self.dueling_type == 'avg':
                x = v + (a - a.mean(1, keepdim=True))
            elif self.dueling_type == 'max':
                x = v + (a - a.max(1, keepdim=True)[0])
            elif self.dueling_type == 'naive':
                x = v + a
            else:
                assert False, "dueling_type must be one of {'avg', 'max', 'naive'}"
            del v_ind_vb, a_ind_vb, v, a
            return x
        else:
            return self.fc4(x.view(x.size(0), -1))


class EmptyModel(Model):

    def __init__(self, args):
        super(EmptyModel, self).__init__(args)
        self._reset()

    def _init_weights(self):
        pass

    def print_model(self):
        self.logger.warning('<-----------------------------------> Model')
        self.logger.warning(self)

    def _reset(self):
        self._init_weights()
        self.type(self.dtype)
        self.print_model()

    def forward(self, input):
        pass

