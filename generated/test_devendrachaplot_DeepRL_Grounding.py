import sys
_module = sys.modules[__name__]
del sys
a3c_main = _module
a3c_test = _module
a3c_train = _module
constants = _module
env = _module
env_test = _module
models = _module
utils = _module
doom = _module
points = _module

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


import numpy as np


import torch


import torch.nn.functional as F


import logging


from torch.autograd import Variable


import torch.optim as optim


import torch.nn as nn


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
    return out


def weights_init(m):
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


class A3C_LSTM_GA(torch.nn.Module):

    def __init__(self, args):
        super(A3C_LSTM_GA, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.gru_hidden_size = 256
        self.input_size = args.input_size
        self.embedding = nn.Embedding(self.input_size, 32)
        self.gru = nn.GRU(32, self.gru_hidden_size)
        self.attn_linear = nn.Linear(self.gru_hidden_size, 64)
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(args.max_episode_length + 1,
            self.time_emb_dim)
        self.linear = nn.Linear(64 * 8 * 17, 256)
        self.lstm = nn.LSTMCell(256, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 3)
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self
            .actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self
            .critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def forward(self, inputs):
        x, input_inst, (tx, hx, cx) = inputs
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_image_rep = F.relu(self.conv3(x))
        encoder_hidden = Variable(torch.zeros(1, 1, self.gru_hidden_size))
        for i in range(input_inst.data.size(1)):
            word_embedding = self.embedding(input_inst[0, i]).unsqueeze(0)
            _, encoder_hidden = self.gru(word_embedding, encoder_hidden)
        x_instr_rep = encoder_hidden.view(encoder_hidden.size(1), -1)
        x_attention = F.sigmoid(self.attn_linear(x_instr_rep))
        x_attention = x_attention.unsqueeze(2).unsqueeze(3)
        x_attention = x_attention.expand(1, 64, 8, 17)
        assert x_image_rep.size() == x_attention.size()
        x = x_image_rep * x_attention
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear(x))
        hx, cx = self.lstm(x, (hx, cx))
        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_devendrachaplot_DeepRL_Grounding(_paritybench_base):
    pass
